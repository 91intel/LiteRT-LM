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

"""LiteRT LM model runner for LM Eval Harness."""

from typing import List, Tuple

try:
  import litert_lm  # pytype: disable=g-import-not-at-top
except ImportError:
  import litert_lm  # pylint: disable=g-import-not-at-top

try:
  from lm_eval.api.model import LM  # type: ignore  # pytype: disable=g-import-not-at-top,g-importing-member
  from lm_eval.api.registry import register_model  # type: ignore  # pytype: disable=g-import-not-at-top,g-importing-member
except ImportError:
  # Fallback to allow module import when lm_eval is pip-installed locally
  class LM:
    pass

  def register_model(*args):
    return lambda x: x


# Map string backend to litert_lm.Backend enum.
_BACKEND_MAP = {
    "CPU": litert_lm.Backend.CPU,
    "GPU": litert_lm.Backend.GPU,
}


@register_model("litert_lm")
class LitertLmModelRunner(LM):
  """A wrapper for the LiteRT LM model to be used with the LM Eval Harness."""

  def __init__(
      self,
      model_path: str,
      batch_size: int = 1,
      backend: str = "CPU",
      max_num_tokens: int = 4096,
      **kwargs
  ):
    super().__init__()
    self.model_path = model_path
    self.batch_size = int(batch_size)

    self.backend = _BACKEND_MAP.get(backend.upper(), litert_lm.Backend.CPU)
    self.max_num_tokens = int(max_num_tokens)

    self.engine = litert_lm.Engine(
        model_path=self.model_path,
        backend=self.backend,
        max_num_tokens=self.max_num_tokens,
        cache_dir=":nocache" if kwargs.get("no_cache", False) else "",
    )
    self.engine.__enter__()

  def __del__(self):
    if hasattr(self, "engine"):
      try:
        self.engine.__exit__(None, None, None)
      except Exception:  # pylint: disable=broad-except
        pass

  def generate_until(self, requests) -> List[str]:
    res = []
    for request in requests:
      args = request.args if hasattr(request, "args") else request
      context, gen_args = args

      until = gen_args.get("until")
      if until and not isinstance(until, list):
        until = [until]

      with self.engine.create_conversation() as conversation:
        response = conversation.send_message(context)
        content_list = response.get("content", [])
        text_response = ""
        for item in content_list:
          if item.get("type") == "text":
            text_response += item.get("text", "")

        if until:
          for stop_seq in until:
            if stop_seq in text_response:
              text_response = text_response.split(stop_seq)[0]
              break
        res.append(text_response)
    return res

  def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
    res = []
    for request in requests:
      args = request.args if hasattr(request, "args") else request
      context, continuation = args

      with self.engine.create_session() as session:
        session.run_prefill([context])
        scoring_responses = session.run_text_scoring([continuation])
        score = scoring_responses.scores[0] if scoring_responses.scores else 0.0
        res.append(
            (score, True)
        )  # is_greedy=True as placeholder, need to be implemented
    return res

  def loglikelihood_rolling(self, requests) -> List[Tuple[float]]:
    # Pending to expose per-token logprobs
    raise NotImplementedError()
