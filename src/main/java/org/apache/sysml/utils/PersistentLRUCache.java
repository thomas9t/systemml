/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ref.SoftReference;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.FastBufferedDataInputStream;
import org.apache.sysml.runtime.util.FastBufferedDataOutputStream;

/**
 * Simple utility to store double[], float[] and MatrixBlock in-memory.
 * When memory is full or if capacity is exceeded, SimplePersistingCache stores the least recently used values into the local filesystem.
 * Assumption: GC occurs before an OutOfMemoryException, and GC requires prior finalize call.
 * 
 * The user should use custom put and get methods:
 * - void put(String key, double[] value);
 * - void put(String key, float[] value);
 * - void put(String key, MatrixBlock value);
 * - double [] getAsDoubleArray(String key);
 * - float [] getAsFloatArray(String key);
 * - MatrixBlock getAsMatrixBlock(String key);
 * 
 * Additionally, the user can also use standard Map methods:
 * - void clear();
 * - boolean containsKey(String key)
 * - remove(String key);
 */
public class PersistentLRUCache extends LinkedHashMap<String, ValueWrapper> {
	static final Log LOG = LogFactory.getLog(PersistentLRUCache.class.getName());
	private static final long serialVersionUID = -6838798881747433047L;
	private final String _prefixFilePath;
	private final AtomicLong _currentNumBytes = new AtomicLong();
	private final long _maxNumBytes;
	
	/**
	 * Creates a persisting cache
	 * @param maxNumBytes maximum capacity in bytes
	 * @throws IOException if unable to create a temporary directory on the local file system
	 */
	public PersistentLRUCache(long maxNumBytes) throws IOException {
		_maxNumBytes = maxNumBytes;
		Random rand = new Random();
		File tmp = Files.createTempDirectory("systemml_" + Math.abs(rand.nextLong())).toFile();
		tmp.deleteOnExit();
		_prefixFilePath = tmp.getAbsolutePath();
	}
	public void put(String key, double[] value) throws FileNotFoundException, IOException {
		ensureCapacity(value.length*Double.BYTES);
		this.put(key, new ValueWrapper(new DataWrapper(key, value, this)));
	}
	public void put(String key, float[] value) throws FileNotFoundException, IOException {
		ensureCapacity(value.length*Float.BYTES);
		this.put(key, new ValueWrapper(new DataWrapper(key, value, this)));
	}
	public void put(String key, MatrixBlock value) throws FileNotFoundException, IOException {
		ensureCapacity(value.getInMemorySize());
		this.put(key, new ValueWrapper(new DataWrapper(key, value, this)));
	}
	
	Map.Entry<String, ValueWrapper> _eldest;
	@Override
    protected boolean removeEldestEntry(Map.Entry<String, ValueWrapper> eldest) {
		_eldest = eldest;
		return false; // Never ask LinkedHashMap to remove eldest entry, instead do that in ensureCapacity.
    }
	
	void ensureCapacity(long newNumBytes) throws FileNotFoundException, IOException {
		if(newNumBytes > _maxNumBytes) {
			throw new DMLRuntimeException("Exceeds maximum capacity. Cannot put a value of size " + newNumBytes + 
					" bytes as max capacity is " + _maxNumBytes + " bytes.");
		}
		long newCapacity = _currentNumBytes.addAndGet(newNumBytes);
		if(newCapacity > _maxNumBytes) {
			synchronized(this) {
				Random rand = new Random();
				String dummyKey = "RAND_KEY_" + Math.abs(rand.nextLong()) + "_" + Math.abs(rand.nextLong());
				ValueWrapper dummyValue = new ValueWrapper(new DataWrapper(dummyKey, new float[] {0}, this));
				int maxIter = size();
				while(_currentNumBytes.get() > _maxNumBytes && maxIter > 0) {
					put(dummyKey, dummyValue); // This will invoke removeEldestEntry, which will set _eldest
					remove(dummyKey);
					if(_eldest != null && _eldest.getKey() != dummyKey) {
						DataWrapper data = _eldest.getValue().get();
						if(data != null) {
							data.write(); // Write the eldest entry to disk if not garbage collected.
						}
						get(_eldest.getKey()); // Make eldest younger.
					}
					maxIter--;
				}
			}
		}
	}
	
//	public void put(String key, MatrixObject value) {
//		_globalMap.put(key, new ValueWrapper(new DataWrapper(key, value, this)));
//	}
	
	String getFilePath(String key) {
		return _prefixFilePath + File.separator + key;
	}
	
	public double [] getAsDoubleArray(String key) throws FileNotFoundException, IOException {
		ValueWrapper value = get(key);
		if(!value.isAvailable()) {
			// Fine-grained synchronization: only one read per key, but will allow parallel loading
			// of distinct keys.
			synchronized(value._lock) {
				if(!value.isAvailable()) {
					value.update(DataWrapper.loadDoubleArr(key, this));
				}
			}
		}
		DataWrapper ret = value.get();
		if(ret == null)
			throw new DMLRuntimeException("Potential race-condition with Java's garbage collector while loading the value in SimplePersistingCache.");
		return ret._dArr;
	}
	
	public float [] getAsFloatArray(String key) throws FileNotFoundException, IOException {
		ValueWrapper value = get(key);
		if(!value.isAvailable()) {
			// Fine-grained synchronization: only one read per key, but will allow parallel loading
			// of distinct keys.
			synchronized(value._lock) {
				if(!value.isAvailable()) {
					value.update(DataWrapper.loadFloatArr(key, this));
				}
			}
		}
		DataWrapper ret = value.get();
		if(ret == null)
			throw new DMLRuntimeException("Potential race-condition with Java's garbage collector while loading the value in SimplePersistingCache.");
		return ret._fArr;
	}
	
	public MatrixBlock getAsMatrixBlock(String key) throws FileNotFoundException, IOException {
		ValueWrapper value = get(key);
		if(!value.isAvailable()) {
			// Fine-grained synchronization: only one read per key, but will allow parallel loading
			// of distinct keys.
			synchronized(value._lock) {
				if(!value.isAvailable()) {
					value.update(DataWrapper.loadMatrixBlock(key, this));
				}
			}
		}
		DataWrapper ret = value.get();
		if(ret == null)
			throw new DMLRuntimeException("Potential race-condition with Java's garbage collector while loading the value in SimplePersistingCache.");
		return ret._mb;
	}
}

// ----------------------------------------------------------------------------------------
// Internal helper class
class DataWrapper {
	final double [] _dArr;
	final float [] _fArr;
	final MatrixBlock _mb;
	final MatrixObject _mo;
	final PersistentLRUCache _cache;
	final String _key;
	DataWrapper(String key, double [] value, PersistentLRUCache cache) {
		_key = key;
		_dArr = value;
		_fArr = null;
		_mb = null;
		_mo = null;
		_cache = cache;
	}
	DataWrapper(String key, float [] value, PersistentLRUCache cache) {
		_key = key;
		_dArr = null;
		_fArr = value;
		_mb = null;
		_mo = null;
		_cache = cache;
	}
	DataWrapper(String key, MatrixBlock value, PersistentLRUCache cache) {
		_key = key;
		_dArr = null;
		_fArr = null;
		_mb = value;
		_mo = null;
		_cache = cache;
	}
	DataWrapper(String key, MatrixObject value, PersistentLRUCache cache) {
		_key = key;
		_dArr = null;
		_fArr = null;
		_mb = null;
		_mo = value;
		_cache = cache;
	}
	@Override
	protected void finalize() throws Throwable {
		super.finalize();
		write();
	}
	
	public void write() throws FileNotFoundException, IOException {
		if(PersistentLRUCache.LOG.isDebugEnabled())
			PersistentLRUCache.LOG.debug("Writing value for the key " + _key + " to disk.");
		if(_dArr != null) {
			try (ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(_cache.getFilePath(_key)))) {
				os.writeInt(_dArr.length);
				for(int i = 0; i < _dArr.length; i++) {
					os.writeDouble(_dArr[i]);
				}
			}
		}
		else if(_fArr != null) {
			try (ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(_cache.getFilePath(_key)))) {
				os.writeInt(_fArr.length);
				for(int i = 0; i < _fArr.length; i++) {
					os.writeFloat(_fArr[i]);
				}
			}
		}
		else if(_mb != null) {
			try(FastBufferedDataOutputStream os = new FastBufferedDataOutputStream(new ObjectOutputStream(new FileOutputStream(_cache.getFilePath(_key))))) {
				os.writeLong(_mb.getInMemorySize());
				_mb.write(os);
			}
		}
		else if(_mo != null) {
			throw new DMLRuntimeException("Not implemented");
		}
		else {
			throw new DMLRuntimeException("Unsupported value type in SimplePersistingCache.");
		}
	}
	
	static DataWrapper loadDoubleArr(String key, PersistentLRUCache cache) throws FileNotFoundException, IOException {
		if(PersistentLRUCache.LOG.isDebugEnabled())
			PersistentLRUCache.LOG.debug("Loading double array the key " + key + " from the disk.");
		double [] ret;
		try (ObjectInputStream is = new ObjectInputStream(new FileInputStream(cache.getFilePath(key)))) {
			int size = is.readInt();
			cache.ensureCapacity(size*Double.BYTES);
			ret = new double[size];
			for(int i = 0; i < size; i++) {
				ret[i] = is.readDouble();
			}
		}
		return new DataWrapper(key, ret, cache);
	}
	
	static DataWrapper loadFloatArr(String key, PersistentLRUCache cache) throws FileNotFoundException, IOException {
		if(PersistentLRUCache.LOG.isDebugEnabled())
			PersistentLRUCache.LOG.debug("Loading float array the key " + key + " from the disk.");
		float [] ret;
		try (ObjectInputStream is = new ObjectInputStream(new FileInputStream(cache.getFilePath(key)))) {
			int size = is.readInt();
			cache.ensureCapacity(size*Float.BYTES);
			ret = new float[size];
			for(int i = 0; i < size; i++) {
				ret[i] = is.readFloat();
			}
		}
		return new DataWrapper(key, ret, cache);
	}
	
	static DataWrapper loadMatrixBlock(String key, PersistentLRUCache cache) throws FileNotFoundException, IOException {
		if(PersistentLRUCache.LOG.isDebugEnabled())
			PersistentLRUCache.LOG.debug("Loading matrix block array the key " + key + " from the disk.");
		MatrixBlock ret;
		try (FastBufferedDataInputStream is = new FastBufferedDataInputStream(new ObjectInputStream(new FileInputStream(cache.getFilePath(key))))) {
			long size = is.readLong();
			cache.ensureCapacity(size);
			ret = new MatrixBlock();
			ret.readFields(is);
		}
		return new DataWrapper(key, ret, cache);
	}
	
}

// Internal helper class
class ValueWrapper {
	final Object _lock;
	private SoftReference<DataWrapper> _ref;
	
	ValueWrapper(DataWrapper _data) {
		_lock = new Object();
		_ref = new SoftReference<>(_data);
	}
	void update(DataWrapper _data) {
		_ref = new SoftReference<>(_data);
	}
	boolean isAvailable() {
		return _ref.get() != null;
	}
	DataWrapper get() {
		return _ref.get();
	}
}

