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

constexpr static std::size_t huge_page_size = 1 << 21; // 2 MiB

JNIEXPORT jlong JNICALL Java_ai_rapids_cudf_THP_allocate(
    JNIEnv *env, jclass, jlong len) {
  void *p = nullptr;
  posix_memalign(&p, huge_page_size, len);
  madvise(p, len, MADV_HUGEPAGE);
  if (p == nullptr) {
    return 0;
  } else {
    return dynamic_cast<long>(p);
  }
}

} // extern "C"
