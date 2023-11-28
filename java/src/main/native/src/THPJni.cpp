/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <errno.h>
#include <fcntl.h>
#include <jni.h>
#include <string.h>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/types.h>

#include "jni_utils.hpp"

extern "C" {


JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_THP_allocate(
    JNIEnv *env, jclass, jlong jlen) {
  const std::size_t huge_page_limit = 1 << 21; // 2 MiB
  const std::size_t huge_page_size = sysconf(_SC_PAGESIZE);
  void *p = nullptr;
  const std::size_t len = static_cast<std::size_t>(jlen);
  if (len >= huge_page_limit) {
    char* disable_it = getenv("DISABLE_THP_MADVISE");
    posix_memalign(&p, huge_page_size, len);
    if (disable_it == nullptr || disable_it[0] == '0') {
      madvise(p, len, MADV_HUGEPAGE);
    }
  }
  if (p == nullptr) {
    return 0;
  } else {
    return reinterpret_cast<long>(p);
  }
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_THP_free(
    JNIEnv *env, jclass, jlong jaddr, jlong) {
  void* addr = reinterpret_cast<void*>(jaddr);
  std::free(addr);
}

JNIEXPORT void JNICALL Java_ai_rapids_cudf_THP_copyMemoryNative(
    JNIEnv *env, jclass, jbyteArray src, jlong jsrcAddr, 
    jbyteArray dst, jlong jdstAddr, jlong length) {
  void* dstAddr = reinterpret_cast<void*>(jdstAddr);
  void* srcAddr = reinterpret_cast<void*>(jsrcAddr);
  jbyte* jSrcArray = nullptr;
  jbyte* jDstArray = nullptr;

  if (src != nullptr) {
    jSrcArray = env->GetByteArrayElements(src, NULL);
    srcAddr = reinterpret_cast<void*>(
      reinterpret_cast<long>(jSrcArray) + reinterpret_cast<long>(srcAddr));
  }

  if (dst != nullptr) {
    jDstArray = env->GetByteArrayElements(dst, NULL);
    dstAddr = reinterpret_cast<void*>(
      reinterpret_cast<long>(jDstArray) + reinterpret_cast<long>(dstAddr));
  }

  memcpy(dstAddr, srcAddr, length);

  if (jSrcArray != nullptr) {
    env->ReleaseByteArrayElements(src, jSrcArray, 0);
  }
  if (jDstArray != nullptr) {
    env->ReleaseByteArrayElements(dst, jDstArray, 0);
  }
}
} // extern "C"
