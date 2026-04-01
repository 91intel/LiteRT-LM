# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

py_library(
    name = "init",
    srcs = ["__init__.py"],
    strict_deps = False,
)

py_library(
    name = "litert_lm_model",
    srcs = ["litert_lm_model.py"],
    imports = ["."],
    strict_deps = False,
    deps = [
        ":init",
        "//python/litert_lm",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    strict_deps = False,
    deps = [":init"],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    strict_deps = False,
    deps = [
        ":init",
        ":utils",
        "@absl_py//absl/testing:absltest",
    ],
)

py_library(
    name = "litert_lm_eval_lib",
    srcs = ["litert_lm_eval.py"],
    imports = ["."],
    strict_deps = False,
    deps = [
        ":init",
        ":litert_lm_model",
        ":utils",
        "//third_party/py/lm_eval",
    ],
)

py_binary(
    name = "litert_lm_eval",
    srcs = ["litert_lm_eval.py"],
    strict_deps = False,
    deps = [
        ":litert_lm_eval_lib",
    ],
)

py_test(
    name = "litert_lm_eval_test",
    srcs = ["litert_lm_eval_test.py"],
    strict_deps = False,
    deps = [
        ":init",
        ":litert_lm_eval_lib",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
