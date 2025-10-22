/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jni.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <stdexcept>

extern "C" {

// Helper function to throw IndexOutOfBoundsException
static void throw_index_out_of_bounds(JNIEnv* env, const char* message) {
  jclass exception_class = env->FindClass("java/lang/IndexOutOfBoundsException");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message);
  }
}

// Helper function to throw OutOfMemoryError
static void throw_out_of_memory(JNIEnv* env, const char* message) {
  jclass exception_class = env->FindClass("java/lang/OutOfMemoryError");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message);
  }
}

// Get system page size
JNIEXPORT jint JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_pageSize(JNIEnv* env, jclass) {
  return static_cast<jint>(sysconf(_SC_PAGESIZE));
}

// Allocate memory
JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_allocate(JNIEnv* env, jclass, jlong bytes) {
  if (bytes <= 0) {
    throw_index_out_of_bounds(env, "Cannot allocate non-positive number of bytes");
    return 0;
  }
  
  void* ptr = std::malloc(static_cast<size_t>(bytes));
  if (ptr == nullptr) {
    throw_out_of_memory(env, "Failed to allocate memory");
    return 0;
  }
  
  return reinterpret_cast<jlong>(ptr);
}

// Free memory
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_free(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    return; // Freeing null is a no-op
  }
  std::free(reinterpret_cast<void*>(address));
}

// Set memory to a value
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setMemory(JNIEnv* env, jclass, 
                                                jlong address, jlong size, jbyte value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  if (size < 0) {
    throw_index_out_of_bounds(env, "Negative size");
    return;
  }
  
  std::memset(reinterpret_cast<void*>(address), static_cast<int>(value), 
              static_cast<size_t>(size));
}

// Byte operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setByte(JNIEnv* env, jclass, jlong address, jbyte value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jbyte*>(address) = value;
}

JNIEXPORT jbyte JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getByte(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0;
  }
  return *reinterpret_cast<jbyte*>(address);
}

// Int operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setInt(JNIEnv* env, jclass, jlong address, jint value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jint*>(address) = value;
}

JNIEXPORT jint JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getInt(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0;
  }
  return *reinterpret_cast<jint*>(address);
}

// Long operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setLong(JNIEnv* env, jclass, jlong address, jlong value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jlong*>(address) = value;
}

JNIEXPORT jlong JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getLong(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0;
  }
  return *reinterpret_cast<jlong*>(address);
}

// Short operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setShort(JNIEnv* env, jclass, jlong address, jshort value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jshort*>(address) = value;
}

JNIEXPORT jshort JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getShort(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0;
  }
  return *reinterpret_cast<jshort*>(address);
}

// Double operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setDouble(JNIEnv* env, jclass, jlong address, jdouble value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jdouble*>(address) = value;
}

JNIEXPORT jdouble JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getDouble(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0.0;
  }
  return *reinterpret_cast<jdouble*>(address);
}

// Float operations
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_setFloat(JNIEnv* env, jclass, jlong address, jfloat value) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return;
  }
  *reinterpret_cast<jfloat*>(address) = value;
}

JNIEXPORT jfloat JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_getFloat(JNIEnv* env, jclass, jlong address) {
  if (address == 0) {
    throw_index_out_of_bounds(env, "Null pointer access");
    return 0.0f;
  }
  return *reinterpret_cast<jfloat*>(address);
}

// Copy memory - the most complex operation
JNIEXPORT void JNICALL
Java_ai_rapids_cudf_JniMemoryAccessor_copyMemoryNative(JNIEnv* env, jclass,
                                                       jobject src, jlong srcOffset,
                                                       jobject dst, jlong dstOffset,
                                                       jlong length) {
  if (length <= 0) {
    return; // Nothing to copy
  }
  
  void* src_ptr = nullptr;
  void* dst_ptr = nullptr;
  void* src_array_critical = nullptr;
  void* dst_array_critical = nullptr;
  
  try {
    // Handle source
    if (src != nullptr) {
      // Source is a Java array
      src_array_critical = env->GetPrimitiveArrayCritical(static_cast<jarray>(src), nullptr);
      if (src_array_critical == nullptr) {
        throw_out_of_memory(env, "Failed to get source array critical");
        return;
      }
      src_ptr = static_cast<char*>(src_array_critical) + srcOffset;
    } else {
      // Source is native memory
      if (srcOffset == 0) {
        throw_index_out_of_bounds(env, "Null source pointer");
        return;
      }
      src_ptr = reinterpret_cast<void*>(srcOffset);
    }
    
    // Handle destination
    if (dst != nullptr) {
      // Destination is a Java array
      dst_array_critical = env->GetPrimitiveArrayCritical(static_cast<jarray>(dst), nullptr);
      if (dst_array_critical == nullptr) {
        if (src_array_critical != nullptr) {
          env->ReleasePrimitiveArrayCritical(static_cast<jarray>(src), src_array_critical, JNI_ABORT);
        }
        throw_out_of_memory(env, "Failed to get destination array critical");
        return;
      }
      dst_ptr = static_cast<char*>(dst_array_critical) + dstOffset;
    } else {
      // Destination is native memory
      if (dstOffset == 0) {
        if (src_array_critical != nullptr) {
          env->ReleasePrimitiveArrayCritical(static_cast<jarray>(src), src_array_critical, JNI_ABORT);
        }
        throw_index_out_of_bounds(env, "Null destination pointer");
        return;
      }
      dst_ptr = reinterpret_cast<void*>(dstOffset);
    }
    
    // Perform the copy
    std::memmove(dst_ptr, src_ptr, static_cast<size_t>(length));
    
    // Release array critical sections
    if (src_array_critical != nullptr) {
      env->ReleasePrimitiveArrayCritical(static_cast<jarray>(src), src_array_critical, JNI_ABORT);
    }
    if (dst_array_critical != nullptr) {
      env->ReleasePrimitiveArrayCritical(static_cast<jarray>(dst), dst_array_critical, 0);
    }
    
  } catch (const std::exception& e) {
    // Clean up on exception
    if (src_array_critical != nullptr) {
      env->ReleasePrimitiveArrayCritical(static_cast<jarray>(src), src_array_critical, JNI_ABORT);
    }
    if (dst_array_critical != nullptr) {
      env->ReleasePrimitiveArrayCritical(static_cast<jarray>(dst), dst_array_critical, JNI_ABORT);
    }
    
    jclass exception_class = env->FindClass("java/lang/RuntimeException");
    if (exception_class != nullptr) {
      env->ThrowNew(exception_class, e.what());
    }
  }
}

} // extern "C"

