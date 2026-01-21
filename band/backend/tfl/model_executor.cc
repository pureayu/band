// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "band/backend/tfl/model_executor.h"

#include "band/backend/tfl/model.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend/tfl/util.h"
#include "band/common.h"
#include "band/device/cpu.h"
#include "band/logger.h"
#include "band/worker.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/subgraph.h"
#include <dlfcn.h>
#include <iostream>
#ifdef CL_DELEGATE_NO_GL
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/nnapi/nnapi_util.h"
#endif  // __ANDROID__
#include "absl/strings/str_format.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#ifdef BAND_USE_QNN
#include "QNN/TFLiteDelegate/QnnTFLiteDelegate.h"
#endif
namespace band {
namespace tfl {
#ifdef BAND_TRACE_SCHED
#include <cstdio>
#define BAND_TRACEF(...) do { std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)
#else
#define BAND_TRACEF(...) do {} while (0)
#endif
#ifdef BAND_USE_QNN
typedef TfLiteQnnDelegateOptions (*TfLiteQnnDelegateOptionsDefault_Ptr)();
typedef TfLiteDelegate* (*TfLiteQnnDelegateCreate_Ptr)(const TfLiteQnnDelegateOptions* options);
typedef void (*TfLiteQnnDelegateDelete_Ptr)(TfLiteDelegate* delegate);

static void* g_qnn_lib_handle = nullptr;
static TfLiteQnnDelegateDelete_Ptr g_qnn_delete_fn = nullptr;

// 创建 NPU Delegate (QNN)
static TfLiteDelegate* CreateNPUDelegate() {
  if (!g_qnn_lib_handle) {
    // 你原来写死的路径可以保留，最稳（不依赖 LD_LIBRARY_PATH）
    const char* delegate_so = "/data/local/tmp/qnn_delegate/libQnnTFLiteDelegate.so";
    g_qnn_lib_handle = dlopen(delegate_so, RTLD_NOW | RTLD_LOCAL);
    if (!g_qnn_lib_handle) {
      std::cerr << "[Band][QNN] dlopen failed: " << delegate_so
                << " err=" << dlerror() << std::endl;
      return nullptr;
    }
  }

  auto qnn_default_fn = (TfLiteQnnDelegateOptionsDefault_Ptr)dlsym(
      g_qnn_lib_handle, "TfLiteQnnDelegateOptionsDefault");
  auto qnn_create_fn = (TfLiteQnnDelegateCreate_Ptr)dlsym(
      g_qnn_lib_handle, "TfLiteQnnDelegateCreate");
  g_qnn_delete_fn = (TfLiteQnnDelegateDelete_Ptr)dlsym(
      g_qnn_lib_handle, "TfLiteQnnDelegateDelete");

  if (!qnn_default_fn || !qnn_create_fn || !g_qnn_delete_fn) {
    std::cerr << "[Band][QNN] dlsym failed: "
              << "default=" << (void*)qnn_default_fn
              << " create=" << (void*)qnn_create_fn
              << " delete=" << (void*)g_qnn_delete_fn
              << " err=" << dlerror() << std::endl;
    return nullptr;
  }

  TfLiteQnnDelegateOptions options = qnn_default_fn();

  // 你原框架的关键配置：HTP + FP16 + 指定后端 so + skel 目录
  options.backend_type = kHtpBackend;
  options.htp_options.precision = kHtpFp16;
  options.library_path = "/data/local/tmp/qnn_delegate/libQnnHtp.so";
  options.skel_library_dir = "/data/local/tmp/qnn_delegate/";

  TfLiteDelegate* d = qnn_create_fn(&options);
  if (!d) {
    std::cerr << "[Band][QNN] TfLiteQnnDelegateCreate returned nullptr" << std::endl;
  }
  return d;
}
#endif  // BAND_USE_QNN

std::map<DeviceFlag, tflite::Interpreter::TfLiteDelegatePtr>
    TfLiteModelExecutor::delegates_ = {};

TfLiteModelExecutor::~TfLiteModelExecutor() {
  // explicitly remove interpreters first
  // since delegates own interpreter.
  interpreters_.clear();
}

absl::StatusOr<ModelSpec> TfLiteModelExecutor::InvestigateModelSpec(
    interface::IModel* model) {
  int num_ops;
  int num_tensors;
  std::vector<DataType> tensor_types;
  std::set<int> input_tensor_indices;
  std::set<int> output_tensor_indices;
  std::vector<std::set<int>> op_input_tensors;
  std::vector<std::set<int>> op_output_tensors;
  std::map<DeviceFlag, std::set<int>> unsupported_ops;
  std::set<DeviceFlag> unavailable_devices;

  // Analyze entire model based on CPU interpereter
  {
    auto status_or_interpreter =
        CreateTfLiteInterpreter(model, DeviceFlag::kCPU);
    if (!status_or_interpreter.ok()) {
      return status_or_interpreter.status();
    }
    std::unique_ptr<tflite::Interpreter>& interpreter =
        status_or_interpreter.value();

    tflite::Subgraph& primary_subgraph = interpreter->primary_subgraph();
    std::vector<int>& execution_plan = primary_subgraph.execution_plan();
    num_ops = execution_plan.size();

    // allocate circular buffer for model IO
    std::vector<TfLiteTensor*> input_tensors;
    std::vector<TfLiteTensor*> output_tensors;

    for (int input_tensor : primary_subgraph.inputs()) {
      input_tensors.push_back(primary_subgraph.tensor(input_tensor));
    }

    for (int output_tensor : primary_subgraph.outputs()) {
      output_tensors.push_back(primary_subgraph.tensor(output_tensor));
    }
    // check input/output/intermediate tensors to fill in
    // model_spec.output_tensors and model_spec.tensor_types
    for (auto node_index : execution_plan) {
      const TfLiteNode& node =
          primary_subgraph.node_and_registration(node_index)->first;

      op_input_tensors.push_back({});
      std::set<int> tensor_indices;
      for (int input_tensor : tflite::TfLiteIntArrayView(node.inputs)) {
        if (input_tensor == kTfLiteOptionalTensor) {
          continue;
        }
        tensor_indices.insert(input_tensor);
        // skip input tensors that are always available
        if (primary_subgraph.tensor(input_tensor)->allocation_type !=
            kTfLiteMmapRo) {
          op_input_tensors.back().insert(input_tensor);
        }
      }

      op_output_tensors.push_back({});
      for (int output_tensor : tflite::TfLiteIntArrayView(node.outputs)) {
        if (output_tensor == kTfLiteOptionalTensor) {
          continue;
        }
        tensor_indices.insert(output_tensor);
        if (primary_subgraph.tensor(output_tensor)->allocation_type !=
            kTfLiteMmapRo) {
          op_output_tensors.back().insert(output_tensor);
        }
      }

      for (auto i : tensor_indices) {
        const auto* tensor = primary_subgraph.tensor(i);
        tensor_types.push_back(GetBandDataType(tensor->type));
      }
    }

    std::copy(
        primary_subgraph.inputs().begin(), primary_subgraph.inputs().end(),
        std::inserter(input_tensor_indices, input_tensor_indices.begin()));

    std::copy(
        primary_subgraph.outputs().begin(), primary_subgraph.outputs().end(),
        std::inserter(output_tensor_indices, output_tensor_indices.begin()));
    num_tensors = primary_subgraph.tensors_size();
  }

  // also check unsupported ops to fill in model_spec.unsupported_ops
  for (size_t i = 0; i < EnumLength<DeviceFlag>(); ++i) {
    DeviceFlag device_flag = static_cast<DeviceFlag>(i);
    unsupported_ops[device_flag] = {};

    if (device_flag == DeviceFlag::kCPU) {
      // no need to check supportability for CPU
      continue;
    }

    auto status_or_interpreter = CreateTfLiteInterpreter(model, device_flag);
    if (!status_or_interpreter.ok()) {
      unavailable_devices.insert(device_flag);
      continue;
    }

    std::unique_ptr<tflite::Interpreter>& interpreter =
        status_or_interpreter.value();
    tflite::Subgraph& primary_subgraph = interpreter->primary_subgraph();
    std::vector<int>& execution_plan = primary_subgraph.execution_plan();

    for (auto node_index : execution_plan) {
      const TfLiteNode& node =
          primary_subgraph.node_and_registration(node_index)->first;
      if (node.delegate == nullptr) {
        // this subgraph is always a 0~num_ops-1 CPU subgraph so
        // the node-->op mapping is basically the identity mapping
        unsupported_ops[device_flag].insert(node_index);
      }
    }
  }

  ModelSpec model_spec(num_ops, num_tensors, tensor_types, input_tensor_indices,
                       output_tensor_indices, op_input_tensors,
                       op_output_tensors, unsupported_ops, unavailable_devices);

  model_spec.path = model->GetPath();
  return model_spec;
}

absl::Status TfLiteModelExecutor::PrepareSubgraph(interface::IModel* model,
                                                  std::set<int> ops,
                                                  std::set<int> unit_indices) {
  if (model_id_ != model->GetId()) {
    return absl::InternalError(
        absl::StrFormat("Failed to prepare subgraph, given model id %d != "
                        "predeclared interpreter's model id %d",
                        model->GetId(), model_id_));
  }

  std::unique_ptr<tflite::Interpreter> interpreter =
      CreateTfLiteInterpreter(model, device_flag_, ops).value();

  if (!interpreter) {
    return absl::InternalError("Failed to create TFLite Interpreter");
  }
  interpreters_[SubgraphKey(model->GetId(), worker_id_, unit_indices)] =
      std::move(interpreter);
  return absl::OkStatus();
}

BackendType TfLiteModelExecutor::GetBackendType() const {
  return BackendType::kTfLite;
}

const std::vector<int>& TfLiteModelExecutor::GetInputs(
    const SubgraphKey& key) const {
  return GetInterpreter(key)->inputs();
}

const std::vector<int>& TfLiteModelExecutor::GetOutputs(
    const SubgraphKey& key) const {
  return GetInterpreter(key)->outputs();
}

const char* TfLiteModelExecutor::GetInputName(const SubgraphKey& key,
                                              int index) const {
  return GetInterpreter(key)->GetInputName(index);
}

const char* TfLiteModelExecutor::GetOutputName(const SubgraphKey& key,
                                               int index) const {
  return GetInterpreter(key)->GetOutputName(index);
}

size_t TfLiteModelExecutor::GetNumTensors(const SubgraphKey& key) const {
  return GetInterpreter(key)->tensors_size();
}

size_t TfLiteModelExecutor::GetNumNodes(const SubgraphKey& key) const {
  return GetInterpreter(key)->nodes_size();
}

std::shared_ptr<interface::ITensorView> TfLiteModelExecutor::GetTensorView(
    const SubgraphKey& key, int index) {
  return std::make_shared<TfLiteTensorView>(GetInterpreter(key)->tensor(index));
}

SubgraphKey TfLiteModelExecutor::GetLargestSubgraphKey() const {
  SubgraphKey largest_key;
  size_t largest_num_ops = 0;

  for (const auto& it : interpreters_) {
    if (largest_num_ops < it.second->nodes_size()) {
      largest_key = it.first;
      largest_num_ops = it.second->nodes_size();
    }
  }

  return largest_key;
}

bool TfLiteModelExecutor::HasSubgraph(const SubgraphKey& key) const {
  return interpreters_.find(key) != interpreters_.end();
}

#include <cstdio>   // 确保有这个

absl::Status TfLiteModelExecutor::ExecuteSubgraph(const SubgraphKey& key) {
  TfLiteDelegate* dptr = nullptr;
  auto st_or = GetDeviceDelegate(device_flag_);
  if (st_or.ok()) dptr = *st_or;

  fprintf(stderr,
          "[EXEC] worker_id=%d exec_device=%s delegate=%p interp=%p\n",
          key.GetWorkerId(),
          ToString(device_flag_),
          (void*)dptr,
          (void*)interpreters_[key].get());

  if (!HasSubgraph(key)) {
    return absl::InternalError("Cannot find subgraph");
  }
  absl::Status status = GetBandStatus(interpreters_[key]->Invoke());
  return status;
}

void TfLiteModelExecutor::ForEachSubgraph(
    std::function<void(const SubgraphKey&)> visitor) {
  for (const auto& interpreter : interpreters_) {
    visitor(interpreter.first);
  }
}

tflite::Interpreter* TfLiteModelExecutor::GetInterpreter(
    const SubgraphKey& key) {
  auto it = interpreters_.find(key);
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

const tflite::Interpreter* TfLiteModelExecutor::GetInterpreter(
    const SubgraphKey& key) const {
  auto it = interpreters_.find(key);
  return it != interpreters_.end() ? it->second.get() : nullptr;
}

// Discard nnapi backend for devices that has direct support
bool IsNNAPIDeviceUseful(std::string name) {
  static const char* const filter_keywords[] = {
      "nnapi-reference",  // CPU
      "gpu",              // Inefficient than GPUDelegate
      "default"};

  for (auto keyword : filter_keywords) {
    if (name.find(keyword) != std::string::npos) return false;
  }

  return true;
}

DeviceFlag GetNNAPIDeviceFlag(std::string name) {
  auto contains_keywords = [&name](std::vector<std::string> keywords) {
    for (auto keyword : keywords) {
      if (name.find(keyword) != std::string::npos) return true;
    }
    return false;
  };

  if (contains_keywords({"gpu"})) {
    return DeviceFlag::kGPU;
  }

  if (contains_keywords({"dsp"})) {
    return DeviceFlag::kDSP;
  }

  if (contains_keywords({
          "google-edgetpu",
          "liteadaptor",  // Huawei (DaVinci NPU)
          "neuron-ann",   // Mediatek APU
          "qti-hta",      // Hexagon tensor accelerator
          "mtk-neuron"    // Mediatek APU
                          // "mtk-mdla" #TODO(#139) - Mediatek APU for half
                          // float
      })) {
    return DeviceFlag::kNPU;
  }

  // TODO #23
  // 1. Add additional NPU / TPU names
  // 2. Is 'hta' belongs to dsp or npu?

  BAND_LOG(LogSeverity::kWarning,
           "Unknown NNAPI device name: %s. Fallback to CPU.", name.c_str());
  return DeviceFlag::kCPU;
}

absl::StatusOr<std::unique_ptr<tflite::Interpreter>>
TfLiteModelExecutor::CreateTfLiteInterpreter(interface::IModel* model,
                                             DeviceFlag device,
                                             std::set<int> op_indices) {
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::shared_ptr<tflite::InterpreterOptions> option =
      std::make_shared<tflite::InterpreterOptions>();
  option->SetTargetNodes(op_indices);

  TfLiteModel* tf_model = static_cast<TfLiteModel*>(model);
  if (!IsCompatible(model) || !tf_model || !tf_model->IsInitialized()) {
    return nullptr;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*tf_model->GetFlatBufferModel(), resolver,
                                     option.get());
  auto status_or_delegate = GetDeviceDelegate(device);
  if (!status_or_delegate.ok()) {
    return status_or_delegate.status();
  }
  auto delegate = status_or_delegate.value();
  if(device  == DeviceFlag::kCPU){
    //do nothing
  }else{
    if (!delegate) {
    return absl::InternalError(absl::StrFormat(
          "Failed to create Tensorflow Lite delegate for %s", ToString(device)));
    }
    builder.AddDelegate(delegate);
  }
  // if ((device != DeviceFlag::kCPU) && !delegate) {
  //   return absl::InternalError(absl::StrFormat(
  //       "Failed to create Tensorflow Lite delegate for %s", ToString(device)));
  // } else {
  //   builder.AddDelegate(delegate);
  // }

  builder.SetNumThreads(num_threads_);
  if (thread_affinity_mask_.GetMaskBitsVector().size() > 0) {
    builder.SetCpuMasks(thread_affinity_mask_.GetMaskBitsVector());
  }

  if (builder(&interpreter) != kTfLiteOk) {
    return absl::InternalError(
        absl::StrFormat("Failed to build Tensorflow Lite interpreter for %s",
                        ToString(device)));
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError(
        absl::StrFormat("Failed to build Tensorflow Lite interpreter for %s",
                        ToString(device)));
  }
  return std::move(interpreter);
}

absl::StatusOr<TfLiteDelegate*> TfLiteModelExecutor::GetDeviceDelegate(
    DeviceFlag device) {
  auto delegate_it = delegates_.find(device);
  if (delegate_it != delegates_.end()) {
    return delegate_it->second.get();
  }

  tflite::Interpreter::TfLiteDelegatePtr target_delegate =
      tflite::Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});

  std::vector<const char*> string_device_names_list;

  switch (device) {
    #ifdef BAND_TRACE_SCHED
    BAND_TRACEF("[TRACE][GetDeviceDelegate] request device=%s (%d)",
                ToString(device), (int)device);
    #endif
    case DeviceFlag::kCPU: {
      #ifdef BAND_TRACE_SCHED
      BAND_TRACEF("[TRACE][GetDeviceDelegate] CPU path -> nullptr delegate (expected)");
      #endif
      // CPU path intentionally uses nullptr delegate.
      // IMPORTANT: treat as success upstream (do not cache).
      return nullptr;
    }

  #if defined(__ANDROID__)
    case DeviceFlag::kGPU: {
      fprintf(stderr, "[DBG][GetDeviceDelegate] Enter GPU (__ANDROID__)\n");

      TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
      gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      gpu_opts.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      gpu_opts.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
      gpu_opts.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

      // Prevent it from being defaulted to 1 (cf. #34)
      gpu_opts.max_delegated_partitions = 100;

      target_delegate = tflite::Interpreter::TfLiteDelegatePtr(
          TfLiteGpuDelegateV2Create(&gpu_opts), &TfLiteGpuDelegateV2Delete);

      BAND_LOG(LogSeverity::kInfo, "Create Tensorflow Lite GPU delegate");
      fprintf(stderr, "[DBG][GetDeviceDelegate] GPU delegate=%p\n",
              target_delegate.get());
      break;
    }

    case DeviceFlag::kDSP: {
      string_device_names_list = tflite::nnapi::GetDeviceNamesList();

      for (const char* device_name : string_device_names_list) {
        if (!IsNNAPIDeviceUseful(device_name)) continue;

        BAND_LOG(LogSeverity::kInfo, "Available NNAPI device name %s", device_name);

        tflite::StatefulNnApiDelegate::Options nnapi_options =
            tflite::StatefulNnApiDelegate::Options();
        nnapi_options.max_number_delegated_partitions = 0;
        nnapi_options.accelerator_name = device_name;

        tflite::Interpreter::TfLiteDelegatePtr nnapi_delegate =
            tflite::Interpreter::TfLiteDelegatePtr(
                new tflite::StatefulNnApiDelegate(nnapi_options),
                [](TfLiteDelegate* delegate) {
                  delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate);
                });

        if (!nnapi_delegate.get()) continue;

        auto opts = tflite::StatefulNnApiDelegate::GetOptions(nnapi_delegate.get());
        if (GetNNAPIDeviceFlag(opts.accelerator_name) == DeviceFlag::kDSP) {
          target_delegate = std::move(nnapi_delegate);
          BAND_LOG(LogSeverity::kInfo,
                   "Create Tensorflow Lite NNAPI delegate (%s , %s)",
                   opts.accelerator_name, ToString(device));
          break;
        }
      }
      break;
    }

    case DeviceFlag::kNPU: {
#ifdef BAND_TRACE_SCHED
  int qnn_enabled = 0;
#ifdef BAND_USE_QNN
  qnn_enabled = 1;
#endif
  BAND_TRACEF("[TRACE][GetDeviceDelegate] Enter NPU (QNN=%d)", qnn_enabled);
#endif  // BAND_TRACE_SCHED

#ifdef BAND_USE_QNN
  // ===== QNN path =====
  TfLiteDelegate* qnn_delegate_raw = CreateNPUDelegate();

  // 建议：这里用真正的 delete（否则会泄漏 delegate）
  target_delegate = tflite::Interpreter::TfLiteDelegatePtr(
      qnn_delegate_raw,
      [](TfLiteDelegate* d) {
        // 如果你已经在别处保存了 g_qnn_delete_fn，就用它
        if (d && g_qnn_delete_fn) g_qnn_delete_fn(d);
        // 如果你不想依赖 g_qnn_delete_fn，至少别留空 deleter（会泄漏）
      });

  if (target_delegate.get()) {
    BAND_LOG(LogSeverity::kInfo,
             "Create Tensorflow Lite QNN delegate (NPU/HTP)");
  } else {
    // 这里最好给个更明确的错误，方便你定位为何 CreateNPUDelegate 返回 nullptr
    return absl::InternalError("Failed to create QNN delegate for NPU");
  }

  break;

#else
  // ===== NNAPI path (original) =====
  string_device_names_list = tflite::nnapi::GetDeviceNamesList();

  for (const char* device_name : string_device_names_list) {
    if (!IsNNAPIDeviceUseful(device_name)) continue;

    BAND_LOG(LogSeverity::kInfo, "Available NNAPI device name %s", device_name);

    tflite::StatefulNnApiDelegate::Options nnapi_options =
        tflite::StatefulNnApiDelegate::Options();
    nnapi_options.max_number_delegated_partitions = 0;
    nnapi_options.accelerator_name = device_name;

    tflite::Interpreter::TfLiteDelegatePtr nnapi_delegate =
        tflite::Interpreter::TfLiteDelegatePtr(
            new tflite::StatefulNnApiDelegate(nnapi_options),
            [](TfLiteDelegate* delegate) {
              delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate);
            });

    if (!nnapi_delegate.get()) continue;

    auto opts =
        tflite::StatefulNnApiDelegate::GetOptions(nnapi_delegate.get());
    if (GetNNAPIDeviceFlag(opts.accelerator_name) == DeviceFlag::kNPU) {
      target_delegate = std::move(nnapi_delegate);
      BAND_LOG(LogSeverity::kInfo,
               "Create Tensorflow Lite NNAPI delegate (%s , %s)",
               opts.accelerator_name, ToString(device));
      break;
    }
  }

  break;
#endif  // BAND_USE_QNN
}

#else
    case DeviceFlag::kGPU:
    case DeviceFlag::kDSP:
    case DeviceFlag::kNPU: {
      return absl::InternalError(
          "Non-Android build does not support GPU/DSP/NPU delegates");
    }
#endif  // defined(__ANDROID__)

    default: {
      return absl::InternalError(
          absl::StrFormat("Unsupported device type %d",
                          static_cast<size_t>(device)));
    }
  }  // switch(device)

  // If we reach here, CPU has already returned. For others, validate.
  if (!target_delegate.get()) {
    return absl::InternalError("Failed to create delegate");
  }

  // Cache it.
  delegates_.insert({device, std::move(target_delegate)});
  return delegates_.at(device).get();
}


}  // namespace tfl
}  // namespace band