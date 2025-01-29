/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.question_answering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.SentenceTransformerTranslator;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import lombok.extern.log4j.Log4j2;

@Log4j2
public class MultiSpanQuestionAnsweringTranslator extends SentenceTransformerTranslator {
    private List<String> tokens;

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        log.info("Processing input for MultiSpanQuestionAnsweringTranslator");
        NDManager manager = ctx.getNDManager();
        String question = input.getAsString(0);
        String context = input.getAsString(1);
        NDList ndList = new NDList();

        // Tokenize input using BERT tokenizer
        Encoding encodings = tokenizer.encode(question, context);
        tokens = Arrays.asList(encodings.getTokens());
        ctx.setAttachment("encoding", encodings);
        long[] indices = encodings.getIds();
        long[] attentionMask = encodings.getAttentionMask();

        // Create input tensors
        NDArray indicesArray = manager.create(indices);
        indicesArray.setName("input_ids");

        NDArray attentionMaskArray = manager.create(attentionMask);
        attentionMaskArray.setName("attention_mask");

        ndList.add(indicesArray);
        ndList.add(attentionMaskArray);
        return ndList;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        Output output = new Output(200, "OK");
        List<ModelTensor> outputs = new ArrayList<>();

        NDArray numSpans = list.get(0);
        NDArray labels = list.get(1);
        long[] labelValues = labels.toLongArray();

        // Get the encoding from context to find [SEP] position
        Encoding encodings = (Encoding) ctx.getAttachment("encoding");
        int sepPosition = -1;
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens.get(i).equals("[SEP]")) {
                sepPosition = i;
                break;
            }
        }

        // Only process tokens after [SEP]
        List<String> answers = new ArrayList<>();
        List<Integer> currentSpan = new ArrayList<>();

        // Start processing after [SEP] token
        for (int i = sepPosition + 1; i < labelValues.length; i++) {
            int label = (int) labelValues[i];
            if (label == 1) {  // Begin new span
                if (!currentSpan.isEmpty()) {
                    String answer = tokenizer
                        .buildSentence(tokens.subList(currentSpan.get(0), currentSpan.get(currentSpan.size() - 1) + 1));
                    answers.add(answer.trim());
                    currentSpan.clear();
                }
                currentSpan.add(i);
            } else if (label == 2 && !currentSpan.isEmpty()) {  // Continue span
                currentSpan.add(i);
            } else if (label == 0 && !currentSpan.isEmpty()) {  // End span
                String answer = tokenizer.buildSentence(tokens.subList(currentSpan.get(0), currentSpan.get(currentSpan.size() - 1) + 1));
                answers.add(answer.trim());
                currentSpan.clear();
            }
        }

        // Add final span if exists
        if (!currentSpan.isEmpty()) {
            String answer = tokenizer.buildSentence(tokens.subList(currentSpan.get(0), currentSpan.get(currentSpan.size() - 1) + 1));
            answers.add(answer.trim());
        }

        // Create tensor outputs
        outputs
            .add(
                new ModelTensor(
                    "num_spans",
                    new Number[] { numSpans.toLongArray()[0] },
                    new long[] { 1 },
                    MLResultDataType.INT64,
                    null,
                    null,
                    null
                )
            );

        outputs
            .add(
                new ModelTensor(
                    "labels",
                    Arrays.stream(labelValues).boxed().toArray(Number[]::new),
                    new long[] { labelValues.length },
                    MLResultDataType.INT64,
                    null,
                    null,
                    null
                )
            );

        outputs.add(new ModelTensor("answers", String.join(" | ", answers)));

        ModelTensors modelTensorOutput = new ModelTensors(outputs);
        output.add(modelTensorOutput.toBytes());
        return output;
    }
}
