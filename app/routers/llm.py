from fastapi import APIRouter
from openai import OpenAI
from app.config import OPENROUTER_API_KEY
from pydantic import BaseModel, Field
import re
import json
import random
import numpy as np
from typing import List, Dict
from collections import Counter, defaultdict
from tabulate import tabulate

class SubtitleSegment(BaseModel):
    segment_id: int
    start: float
    end: float
    n_best: List[str] = Field(default_factory=list)


# 리스트를 직접 받도록 변경
SubtitleRequest = List[SubtitleSegment]


router = APIRouter(
    prefix="/llm",
    tags=["llm"],
)


@router.post("/generate/{prompt}")
async def generate_text(prompt: str):
    print("ddd")
    llm_model_name = "openai/gpt-5"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    response = client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.7,
    )
    generated_text = response.choices[0].message.content
    return {"generated_text": generated_text}


@router.post("/generate2",
             summary="LLM 기반 자막 후처리 생성")
async def generate_subtitle(body: SubtitleRequest):
    # 새로운 세그먼트 기반 입력을 기존 파이프라인에서 사용하는 형태로 변환
    def convert_segments_to_nbest(segments: SubtitleRequest) -> List[Dict]:
        converted = []
        for segment in segments:
            segment_dict = segment.model_dump()
            segment_id = segment_dict["segment_id"]
            n_best_texts = segment_dict.pop("n_best", []) or [""]

            n_best_with_scores = []
            for idx, text in enumerate(n_best_texts):
                synthetic_score = -float(idx)  # 순위를 보존하기 위한 가짜 점수
                n_best_with_scores.append({"text": text, "score": synthetic_score})

            segment_dict["id"] = segment_id
            segment_dict["n-best"] = n_best_with_scores
            converted.append(segment_dict)

        return converted

    test_json = convert_segments_to_nbest(body)

    llm_model_name = "moonshotai/kimi-k2-thinking"

    class LLMPoweredPostProcessor:
        def __init__(self, api_key):
            if not api_key:
                raise ValueError("OpenRouter API 키가 설정되지 않았습니다.")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
            self.ne_datastore = []

        def build_ne_datastore(self, domain_entities: List[str]):
            self.ne_datastore = domain_entities
            print(
                f"✅ 명명된 개체 데이터스토어 구축 완료: {len(self.ne_datastore)}개 엔티티"
            )

        def retrieve_similar_entities(self, text: str, top_k: int = 3) -> List[str]:
            """단순 문자열 포함 기반 검색"""
            if not self.ne_datastore:
                return []

            matches = []
            for entity in self.ne_datastore:
                if text in entity or entity in text:
                    matches.append((entity, len(entity)))

            matches.sort(key=lambda x: -x[1])
            return [m[0] for m in matches[:top_k]]

        # 1번 방법 [선택적 재순위화] - 배치 처리
        def method1_selective_reranking_batch(
            self,
            batch_items: List[Dict],
            domain_prompt: str = "",
            T: float = 2.0,
            alpha: float = 0.35,
        ) -> List[Dict]:
            print("\n" + "=" * 80)
            print("방법 1: 선택적 재순위화 (Selective Reranking) - 배치 처리")
            print("=" * 80)

            # 모든 문장의 n-best를 하나의 프롬프트로 구성
            batch_data = []
            for item in batch_items:
                batch_data.append(
                    {
                        "id": item["id"],
                        "hypotheses": [h["text"] for h in item["n-best"]],
                    }
                )

            system_prompt = f"""
                You are an expert speech recognition evaluator. {domain_prompt}
                For each sentence group, evaluate all hypotheses for naturalness and correctness on a scale of 0.0 to 1.0.
                Return a JSON object with sentence IDs as keys and arrays of scores as values.

                Example output format:
                {{
                    "1": [0.8, 0.7, 0.6, 0.5, 0.4],
                    "2": [0.9, 0.8, 0.7, 0.6, 0.5]
                }}
            """

            prompt = (
                system_prompt
                + "\n\nSentence Groups:\n"
                + json.dumps(batch_data, ensure_ascii=False)
            )

            print("\n--- LLM 입력 프롬프트 ---")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

            try:
                response = self.client.chat.completions.create(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                
                # 디버깅: LLM 원본 응답 출력
                print("\n--- LLM 원본 응답 ---")
                print(content[:1000] if len(content) > 1000 else content)
                print("--- 응답 끝 ---\n")
                
                # JSON 코드 블록 제거 시도 (```json ... ``` 형식)
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
                    content = content.replace("```json", "").replace("```", "").strip()
                
                all_scores = json.loads(content)
            except Exception as e:
                print(f"❌ LLM 점수 파싱 실패: {e}, 0.5점으로 처리합니다.")
                all_scores = {
                    str(item["id"]): [0.5] * len(item["n-best"]) for item in batch_items
                }

            results = []
            for item in batch_items:
                print("\n" + "-" * 80)
                print(f"ID {item['id']} 상세 점수 계산")
                print("-" * 80)

                hypotheses = item["n-best"]
                item_id = str(item["id"])
                llm_scores = all_scores.get(item_id, [0.5] * len(hypotheses))

                asr_logps = np.array([h["score"] for h in hypotheses])
                llm_scores_arr = np.array(llm_scores)

                def combine_with_temp(asr_logps, llm_scores, T=T, alpha=alpha):
                    asr_scaled = asr_logps / T
                    max_scaled_asr = np.max(asr_scaled)
                    asr_prob = np.exp(asr_scaled - max_scaled_asr)
                    sum_asr_prob = asr_prob.sum()
                    if sum_asr_prob == 0:
                        asr_prob = np.ones_like(asr_prob) / len(asr_prob)
                    else:
                        asr_prob = asr_prob / sum_asr_prob
                    combined = (1 - alpha) * asr_prob + alpha * np.array(llm_scores)
                    return combined

                combined_scores = combine_with_temp(asr_logps, llm_scores_arr)

                table_data = []
                scored_hypotheses = []
                for i, hyp in enumerate(hypotheses):
                    asr_prob = (
                        np.exp(asr_logps[i] - np.max(asr_logps))
                        / np.sum(np.exp(asr_logps - np.max(asr_logps)))
                        if np.sum(np.exp(asr_logps - np.max(asr_logps))) != 0
                        else 1.0 / len(hypotheses)
                    )
                    llm_score = llm_scores_arr[i] if i < len(llm_scores_arr) else 0.5
                    combined_score = combined_scores[i]
                    scored_hypotheses.append({"hyp": hyp, "score": combined_score})
                    table_data.append(
                        [
                            i + 1,
                            f"{asr_prob:.3f}",
                            f"{llm_score:.3f}",
                            f"{combined_score:.3f}",
                            hyp["text"],
                        ]
                    )

                print(
                    tabulate(
                        table_data,
                        headers=["후보", "ASR 점수", "LLM 점수", "최종 점수", "내용"],
                        tablefmt="grid",
                    )
                )

                best_hyp = max(scored_hypotheses, key=lambda x: x["score"])["hyp"]
                print(f"\n✅ 최종 선택: {best_hyp['text']}")
                results.append(best_hyp)

            return results

        # 2번 방법 [생성적 재구성] - 배치 처리
        def method2_generative_reconstitution_batch(
            self, batch_items: List[Dict]
        ) -> List[str]:
            print("\n" + "=" * 80)
            print("방법 2: 생성적 재구성 (Generative Reconstitution) - 배치 처리")
            print("=" * 80)

            batch_data = []
            for item in batch_items:
                batch_data.append(
                    {
                        "id": item["id"],
                        "hypotheses": [h["text"] for h in item["n-best"]],
                    }
                )

            progres_prompt = f"""
                You are an expert Korean transcription corrector.
                For each sentence group below, analyze the N-best ASR hypotheses and produce the single most accurate transcription.

                Return a JSON object with sentence IDs as keys and corrected transcriptions as values.

                Example output format:
                {{
                    "1": "corrected sentence 1",
                    "2": "corrected sentence 2"
                }}

                Sentence Groups:
                {json.dumps(batch_data, ensure_ascii=False)}

                Task:
                - Correct pronunciation errors
                - Use words from all candidate sentences
                - Generate the best transcription for each group

                THINK HARD
            """

            print("\n--- LLM 입력 프롬프트 ---")
            print(
                progres_prompt[:500] + "..."
                if len(progres_prompt) > 500
                else progres_prompt
            )

            response = self.client.chat.completions.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": progres_prompt}],
                temperature=0.2,
            )

            try:
                results_dict = json.loads(response.choices[0].message.content.strip())
            except:
                print("❌ JSON 파싱 실패, 1-best 사용")
                results_dict = {
                    str(item["id"]): item["n-best"][0]["text"] for item in batch_items
                }

            results = []
            for item in batch_items:
                print("\n" + "-" * 80)
                print(f"ID {item['id']} 생성 결과")
                print("-" * 80)

                result = results_dict.get(str(item["id"]), item["n-best"][0]["text"])
                print(f"✅ 최종 생성: {result}")
                results.append(result)

            return results

        # 3번 방법 [검색 증강 교정] - 배치 처리
        def method3_retrieval_augmented_correction_batch(
            self, batch_items: List[Dict]
        ) -> List[str]:
            print("\n" + "=" * 80)
            print("방법 3: 검색 증강 교정 (Retrieval-Augmented Correction) - 배치 처리")
            print("=" * 80)

            batch_data = []
            for item in batch_items:
                best_hyp = item["n-best"][0]["text"]
                retrieved_entities = self.retrieve_similar_entities(best_hyp, top_k=5)
                batch_data.append(
                    {
                        "id": item["id"],
                        "best_hypothesis": best_hyp,
                        "other_hypotheses": [h["text"] for h in item["n-best"][1:]],
                        "named_entities": retrieved_entities,
                    }
                )

            darag_prompt = f"""
                Revise the 'Best hypothesis' from an ASR system for each sentence group using information from 'Other hypotheses' and 'Named Entities'.

                Return a JSON object with sentence IDs as keys and revised transcriptions as values.

                Example output format:
                {{
                    "1": "revised sentence 1",
                    "2": "revised sentence 2"
                }}

                Sentence Groups:
                {json.dumps(batch_data, ensure_ascii=False)}

                Task: Revise each best hypothesis by analyzing for common Korean ASR errors.
                Example of error to look for: '오늘 일곱시에' vs '오늘 이곳이에서'.
            """

            print("\n--- LLM 입력 프롬프트 ---")
            print(
                darag_prompt[:500] + "..." if len(darag_prompt) > 500 else darag_prompt
            )

            response = self.client.chat.completions.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": darag_prompt}],
                temperature=0.1,
            )

            try:
                results_dict = json.loads(response.choices[0].message.content.strip())
            except:
                print("❌ JSON 파싱 실패, 1-best 사용")
                results_dict = {
                    str(item["id"]): item["n-best"][0]["text"] for item in batch_items
                }

            results = []
            for item, batch_item in zip(batch_items, batch_data):
                print("\n" + "-" * 80)
                print(f"ID {item['id']} 교정 결과")
                print("-" * 80)

                print(f"--- 검색된 관련 엔티티 ---")
                print(batch_item["named_entities"])

                result = results_dict.get(str(item["id"]), item["n-best"][0]["text"])
                print(f"\n✅ 최종 교정: {result}")
                results.append(result)

            return results

        # 4번 방법 [클로즈 제약 선택] - 배치 처리
        def method4_cloze_nbest_constrained_batch(
            self, batch_items: List[Dict], max_opts_per_blank: int = 3
        ) -> List[str]:
            print("\n" + "=" * 80)
            print("방법 4: 클로즈 제약 선택 (Cloze N-best Constrained) - 배치 처리")
            print("=" * 80)

            batch_cloze_data = []
            batch_assemblers = {}

            for item in batch_items:
                print("\n" + "-" * 80)
                print(f"ID {item['id']} CLOZE 템플릿 생성")
                print("-" * 80)

                cloze_text, blanks, assemble = self._build_cloze_from_nbest(
                    item["n-best"], max_opts_per_blank
                )

                if len(blanks) == 0:
                    print("BLANK가 없음, 1-best 사용")
                    batch_cloze_data.append(
                        {
                            "id": item["id"],
                            "cloze": item["n-best"][0]["text"],
                            "options": {},
                            "no_blanks": True,
                        }
                    )
                else:
                    print(f"CLOZE: {cloze_text}")
                    for bname, opts in blanks.items():
                        opt_str = ", ".join([f"{o['id']}={o['text']}" for o in opts])
                        print(f"{bname}: {opt_str}")

                    batch_cloze_data.append(
                        {
                            "id": item["id"],
                            "cloze": cloze_text,
                            "options": blanks,
                            "no_blanks": False,
                        }
                    )

                batch_assemblers[item["id"]] = assemble

            prompt = {
                "instruction": (
                    "다음은 여러 문장의 ASR N-best 분석 결과입니다. "
                    "각 문장의 BLANK에 대해 제공된 옵션 중 문맥적으로 가장 적절한 것을 선택하세요. "
                    "출력은 JSON만 반환하세요."
                ),
                "sentences": batch_cloze_data,
                "output_format": {
                    "sentence_id": {
                        "answers": {"BLANK_i": "OptionID"},
                        "scores": {"BLANK_i": {"OptionID": 0.0}},
                    }
                },
            }

            print("\n" + "-" * 80)
            print("LLM에 배치 전체 전송")
            print("-" * 80)

            user_msg = (
                "아래 JSON을 읽고 각 문장의 BLANK별 옵션 점수를 0~1로 부여하고, 최종 선택 answers도 포함한 JSON만 출력하세요.\n\n"
                + json.dumps(prompt, ensure_ascii=False)
            )

            try:
                response = self.client.chat.completions.create(
                    model=llm_model_name,
                    messages=[{"role": "user", "content": user_msg}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                all_data = json.loads(content)
            except Exception as e:
                print(f"❌ LLM 응답 파싱 실패: {e}")
                all_data = {}

            results = []
            for item in batch_items:
                print("\n" + "-" * 80)
                print(f"ID {item['id']} 최종 선택")
                print("-" * 80)

                item_id = str(item["id"])
                cloze_item = next(
                    (c for c in batch_cloze_data if c["id"] == item["id"]), None
                )

                if cloze_item and cloze_item.get("no_blanks", False):
                    result = item["n-best"][0]["text"]
                    print(f"✅ 최종 선택: {result} (BLANK 없음)")
                    results.append(result)
                    continue

                data = all_data.get(item_id, {"answers": {}, "scores": {}})
                blanks = cloze_item["options"] if cloze_item else {}

                final_answers = {}
                for bname, opts in blanks.items():
                    raw_scores = {}
                    if "scores" in data and isinstance(
                        data["scores"].get(bname, None), dict
                    ):
                        for o in opts:
                            raw_scores[o["id"]] = float(
                                data["scores"][bname].get(o["id"], 0.0)
                            )
                    else:
                        for o in opts:
                            raw_scores[o["id"]] = 1.0 / len(opts)

                    ids = [o["id"] for o in opts]
                    vec = np.array([raw_scores[i] for i in ids], dtype=np.float32)
                    vec = vec - vec.max()
                    probs = np.exp(vec)
                    probs = (
                        probs / probs.sum()
                        if probs.sum() > 0
                        else np.ones_like(probs) / len(probs)
                    )
                    chosen_id = ids[int(probs.argmax())]
                    final_answers[bname] = chosen_id

                assemble = batch_assemblers[item["id"]]
                result = assemble(final_answers)
                print(f"✅ 최종 선택: {result}")
                results.append(result)

            return results

        # Private 헬퍼 메서드들
        def _simple_tokenize(self, s: str):
            return [t for t in re.findall(r"\w+|[^\w\s]", s, re.UNICODE)]

        def _simple_detok(self, tokens):
            text = " ".join(tokens)
            text = re.sub(r"\s+([,.;:!?])", r"\1", text)
            text = re.sub(r"\(\s+", "(", text)
            text = re.sub(r"\s+\)", ")", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def _lcs(self, a, b):
            m, n = len(a), len(b)
            dp = [[[] for _ in range(n + 1)] for _ in range(m + 1)]
            for i in range(m):
                for j in range(n):
                    if a[i] == b[j]:
                        dp[i + 1][j + 1] = dp[i][j] + [a[i]]
                    else:
                        dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)
            return dp[m][n]

        def _map_common_positions(self, tokens, common):
            pos = []
            i = 0
            for ctok in common:
                found = False
                while i < len(tokens):
                    if tokens[i] == ctok:
                        pos.append(i)
                        i += 1
                        found = True
                        break
                    i += 1
                if not found:
                    return None
            return pos

        def _build_cloze_from_nbest(self, hypotheses, max_opts_per_blank=3):
            token_lists = [self._simple_tokenize(h["text"]) for h in hypotheses]

            common = token_lists[0][:]
            for t in token_lists[1:]:
                common = self._lcs(common, t)

            K = len(common) + 1
            segments_per_hyp = [[] for _ in hypotheses]
            for hi, toks in enumerate(token_lists):
                pos = self._map_common_positions(toks, common)
                if pos is None:
                    segments_per_hyp[hi] = [toks] + [[] for _ in range(K - 1)]
                    continue
                segs = []
                start = 0
                for p in pos:
                    segs.append(toks[start:p])
                    start = p + 1
                segs.append(toks[start:])
                segments_per_hyp[hi] = segs

            blanks = {}
            blank_indices = []
            for seg_idx in range(K):
                span_texts = []
                for hi, hyp in enumerate(hypotheses):
                    seg = segments_per_hyp[hi][seg_idx]
                    span_text = self._simple_detok(seg) if seg else ""
                    span_texts.append(span_text)

                unique = list(dict.fromkeys(span_texts))
                if len(unique) <= 1:
                    blank_indices.append(None)
                    continue

                norm_unique = ["<NULL>" if x == "" else x for x in unique]

                weight = defaultdict(float)
                for u in norm_unique:
                    weight[u] = 0.0
                cnt = Counter(norm_unique)
                for u, c in cnt.items():
                    total_w = 0.0
                    for hi, hyp in enumerate(hypotheses):
                        val = "<NULL>" if span_texts[hi] == "" else span_texts[hi]
                        if val == u:
                            total_w += float(np.exp(hyp["score"]))
                    weight[u] = c * total_w

                ranked = sorted(norm_unique, key=lambda x: weight[x], reverse=True)
                ranked = ranked[:max_opts_per_blank]

                letters = [chr(ord("A") + i) for i in range(len(ranked))]
                random.shuffle(letters)
                options = [
                    {"id": letters[i], "text": ranked[i]} for i in range(len(ranked))
                ]

                bname = f"BLANK_{len(blanks)+1}"
                blanks[bname] = options
                blank_indices.append(seg_idx)

            pieces = []
            for i in range(K):
                if i == 0:
                    if blank_indices[i] is not None:
                        bcount = sum(1 for x in blank_indices[: i + 1] if x is not None)
                        pieces.append(f"[{f'BLANK_{bcount}'}]")
                else:
                    pieces.append(common[i - 1])
                    if blank_indices[i] is not None:
                        bcount = sum(1 for x in blank_indices[: i + 1] if x is not None)
                        pieces.append(f"[{f'BLANK_{bcount}'}]")

            cloze_text = self._simple_detok(pieces)

            def assemble(chosen: dict):
                out = []
                bptr = 0
                for i in range(K):
                    if i > 0:
                        out.append(common[i - 1])
                    if blank_indices[i] is not None:
                        bptr += 1
                        bname = f"BLANK_{bptr}"
                        opts = {o["id"]: o["text"] for o in blanks[bname]}
                        fill = opts.get(chosen.get(bname, None), "")
                        fill = "" if fill == "<NULL>" else fill
                        if fill:
                            out.append(fill)
                return self._simple_detok(out)

            return cloze_text, blanks, assemble

    def format_text_for_table(text: str, max_length: int = 50) -> str:
        """테이블 출력을 위한 텍스트 포맷팅"""
        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    def process_batch(
        llm_processor: LLMPoweredPostProcessor,
        batch_data: List[Dict],
        method: str = "all",
    ) -> List[Dict]:
        """배치 단위로 처리 - 각 방법당 1번의 LLM 호출"""
        print(f"\n🔄 배치 크기: {len(batch_data)}개 문장")

        results = []
        for item in batch_data:
            segment_id = item.get("segment_id", item["id"])
            results.append(
                {
                    "id": item["id"],
                    "segment_id": segment_id,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "original_nbest": item["n-best"],
                    "results": {},
                }
            )

        # 각 방법마다 배치 전체를 한번에 처리
        if method in ["all", "method1"]:
            print("\n📞 방법1 LLM 호출 (배치 전체)")
            method1_results = llm_processor.method1_selective_reranking_batch(
                batch_data
            )
            for i, result in enumerate(results):
                result["results"]["method1"] = method1_results[i]["text"]

        if method in ["all", "method2"]:
            print("\n📞 방법2 LLM 호출 (배치 전체)")
            method2_results = llm_processor.method2_generative_reconstitution_batch(
                batch_data
            )
            for i, result in enumerate(results):
                result["results"]["method2"] = method2_results[i]

        if method in ["all", "method3"]:
            print("\n📞 방법3 LLM 호출 (배치 전체)")
            method3_results = (
                llm_processor.method3_retrieval_augmented_correction_batch(batch_data)
            )
            for i, result in enumerate(results):
                result["results"]["method3"] = method3_results[i]

        if method in ["all", "method4"]:
            print("\n📞 방법4 LLM 호출 (배치 전체)")
            method4_results = llm_processor.method4_cloze_nbest_constrained_batch(
                batch_data
            )
            for i, result in enumerate(results):
                result["results"]["method4"] = method4_results[i]

        return results

    def run_batch_processing(
        test_data: List[Dict], api_key: str, batch_size: int = 5, method: str = "all"
    ):
        """전체 데이터를 배치로 나눠서 처리"""
        llm_processor = LLMPoweredPostProcessor(api_key)

        domain_entities = []
        if domain_entities:
            llm_processor.build_ne_datastore(domain_entities)

        all_results = []

        for i in range(0, len(test_data), batch_size):
            batch = test_data[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(test_data) + batch_size - 1) // batch_size

            print("\n" + "#" * 80)
            print(
                f"# 배치 {batch_num}/{total_batches} 처리 중 (항목 {i+1}~{min(i+batch_size, len(test_data))})"
            )
            print(f"# 이 배치는 방법당 단 1번의 LLM 호출로 처리됩니다")
            print("#" * 80)

            batch_results = process_batch(llm_processor, batch, method)
            all_results.extend(batch_results)

        print("\n" + "=" * 80)
        print("전체 처리 결과 요약")
        print("=" * 80)

        summary_table = []
        for result in all_results:
            row = [
                result.get("segment_id", result["id"]),
                format_text_for_table(result["original_nbest"][0]["text"]),
            ]

            for method_key in ["method1", "method2", "method3", "method4"]:
                if method_key in result["results"]:
                    row.append(format_text_for_table(result["results"][method_key]))

            summary_table.append(row)

        headers = ["Segment ID", "원본 1-best"]
        if all_results and "method1" in all_results[0]["results"]:
            headers.append("방법1")
        if all_results and "method2" in all_results[0]["results"]:
            headers.append("방법2")
        if all_results and "method3" in all_results[0]["results"]:
            headers.append("방법3")
        if all_results and "method4" in all_results[0]["results"]:
            headers.append("방법4")

        print(tabulate(summary_table, headers=headers, tablefmt="grid"))

        # LLM 호출 횟수 계산
        num_batches = (len(test_data) + batch_size - 1) // batch_size
        methods_count = sum(
            [
                method in ["all", "method1"],
                method in ["all", "method2"],
                method in ["all", "method3"],
                method in ["all", "method4"],
            ]
        )
        if method == "all":
            methods_count = 4

        total_llm_calls = num_batches * methods_count
        print(
            f"\n📊 총 LLM 호출 횟수: {total_llm_calls}회 ({num_batches}개 배치 × {methods_count}개 방법)"
        )

        return all_results

    # 실행
    results = run_batch_processing(
        test_data=test_json, api_key=OPENROUTER_API_KEY, batch_size=5, method="method1"
    )

    return {"results": results, "total_items": len(results)}
