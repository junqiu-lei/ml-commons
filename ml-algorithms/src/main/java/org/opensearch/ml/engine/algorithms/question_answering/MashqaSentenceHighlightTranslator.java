package org.opensearch.ml.engine.algorithms.question_answering;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;
import lombok.extern.log4j.Log4j2;
import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.SentenceTransformerTranslator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

@Log4j2
public class MashqaSentenceHighlightTranslator extends SentenceTransformerTranslator {
    private static final Pattern SENTENCE_BOUNDARY = Pattern.compile("[.!?]+\\s+");
    private List<String> sentences;

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        NDManager manager = ctx.getNDManager();
        String question = input.getAsString(0);
        String context = input.getAsString(1);
        
        // Split context into sentences
        sentences = Arrays.asList(SENTENCE_BOUNDARY.split(context.trim()));
        
        // Tokenize input
        Encoding encodings = tokenizer.encode(question, context);
        ctx.setAttachment("encoding", encodings);
        
        long[] indices = encodings.getIds();
        long[] attentionMask = encodings.getAttentionMask();
        
        // Create sentence IDs tensor
        int[] sentenceIds = new int[indices.length];
        Arrays.fill(sentenceIds, -1);
        
        // Map tokens to sentence IDs
        int currentSentence = 0;
        String[] tokens = encodings.getTokens();
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals("[SEP]")) {
                currentSentence = 0;
                continue;
            }
            if (!tokens[i].equals("[CLS]") && !tokens[i].equals("[PAD]")) {
                sentenceIds[i] = currentSentence;
            }
            if (tokens[i].equals(".") || tokens[i].equals("!") || tokens[i].equals("?")) {
                currentSentence++;
            }
        }

        NDArray indicesArray = manager.create(indices);
        NDArray attentionMaskArray = manager.create(attentionMask);
        NDArray sentenceIdsArray = manager.create(sentenceIds);

        indicesArray.setName("input_ids");
        attentionMaskArray.setName("attention_mask");
        sentenceIdsArray.setName("sentence_ids");

        return new NDList(indicesArray, attentionMaskArray, sentenceIdsArray);
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        Output output = new Output(200, "OK");
        List<ModelTensor> outputs = new ArrayList<>();

        NDArray logits = list.get(0);
        float[] probabilities = logits.softmax(-1).get(":,1").toFloatArray();
        
        // Convert probabilities to binary labels (1 for sentences to highlight)
        int[] labels = new int[probabilities.length];
        int numSpans = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > 0.5) {
                labels[i] = 1;
                numSpans++;
            }
        }

        outputs.add(new ModelTensor(
            "num_spans",
            new Number[] { numSpans },
            new long[] { 1 },
            MLResultDataType.INT64,
            null, null, null
        ));

        outputs.add(new ModelTensor(
            "labels",
            Arrays.stream(labels).boxed().toArray(Number[]::new),
            new long[] { labels.length },
            MLResultDataType.INT64,
            null, null, null
        ));

        ModelTensors modelTensorOutput = new ModelTensors(outputs);
        output.add(modelTensorOutput.toBytes());
        return output;
    }
} 