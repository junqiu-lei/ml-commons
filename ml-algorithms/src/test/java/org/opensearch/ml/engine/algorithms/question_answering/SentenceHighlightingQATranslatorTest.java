/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.ml.engine.algorithms.question_answering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.engine.algorithms.question_answering.QAConstants.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.algorithms.question_answering.sentence.DefaultSentenceSegmenter;
import org.opensearch.ml.engine.algorithms.question_answering.sentence.Sentence;
import org.opensearch.ml.engine.algorithms.question_answering.sentence.SentenceSegmenter;

import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslatorContext;
import lombok.extern.log4j.Log4j2;

@Log4j2
public class SentenceHighlightingQATranslatorTest {

    private SentenceHighlightingQATranslator translator;
    private TranslatorContext translatorContext;
    private List<Sentence> sentences;

    @Before
    public void setUp() {
        // Create test data
        String question = "What are the impacts of climate change?";
        String textContext = "Many coastal cities face increased flooding during storms. "
            + "Farmers are experiencing unpredictable growing seasons and crop failures. "
            + "Scientists predict these environmental shifts will continue to accelerate. "
            + "Global temperatures have risen significantly over the past century. "
            + "Polar ice caps are melting at an alarming rate.";

        // Create real sentences using the actual segmenter
        SentenceSegmenter segmenter = new DefaultSentenceSegmenter();
        sentences = segmenter.segment(textContext);

        // Create mocks
        translator = SentenceHighlightingQATranslator.builder().build();
        translatorContext = mock(TranslatorContext.class);
    }

    @Test
    public void testProcessInput_BasicValidation() {
        // This test verifies that the translator is properly initialized
        assertNotNull(translator);
        assertNotNull(translator.getSegmenter());
        assertEquals(DefaultSentenceSegmenter.class, translator.getSegmenter().getClass());

        // The tokenizer will be initialized in prepare() method
        assertNull(translator.getTokenizer());
    }

    @Test
    public void testProcessOutput_WithRelevantSentences() {
        // Mock context with sentences
        when(translatorContext.getAttachment(KEY_SENTENCES)).thenReturn(sentences);
        
        // Mock model output with sentence indices
        NDArray mockOutput = mock(NDArray.class);
        when(mockOutput.getShape()).thenReturn(new Shape(2));
        when(mockOutput.toLongArray()).thenReturn(new long[]{0, 3}); // Indices of relevant sentences
        
        NDList mockList = new NDList(mockOutput);
        
        // Process output
        Output output = translator.processOutput(translatorContext, mockList);
        
        // Verify results
        assertNotNull(output);
        byte[] bytes = output.getData().getAsBytes();
        ModelTensors tensorOutput = ModelTensors.fromBytes(bytes);
        List<ModelTensor> modelTensorsList = tensorOutput.getMlModelTensors();
        
        // Should have one tensor with the highlights
        assertEquals(1, modelTensorsList.size());
        ModelTensor highlightsTensor = modelTensorsList.get(0);
        assertEquals(FIELD_HIGHLIGHTS, highlightsTensor.getName());
        
        // Get the highlights from the dataAsMap
        @SuppressWarnings("unchecked")
        Map<String, Object> dataMap = (Map<String, Object>) highlightsTensor.getDataAsMap();
        assertNotNull(dataMap);
        
        // Should have no error
        assertNull(dataMap.get(FIELD_ERROR));
        
        // Should have highlights
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> highlights = (List<Map<String, Object>>) dataMap.get(FIELD_HIGHLIGHTS);
        assertNotNull(highlights);
        assertEquals(2, highlights.size());
        
        // Verify first highlight
        Map<String, Object> firstHighlight = highlights.get(0);
        assertEquals(0.0, firstHighlight.get(FIELD_POSITION));
        assertEquals("Many coastal cities face increased flooding during storms.", firstHighlight.get(FIELD_TEXT));
        assertEquals(0.0, firstHighlight.get(FIELD_START));
        assertEquals(58.0, firstHighlight.get(FIELD_END));
        
        // Verify second highlight
        Map<String, Object> secondHighlight = highlights.get(1);
        assertEquals(3.0, secondHighlight.get(FIELD_POSITION));
        assertEquals("Global temperatures have risen significantly over the past century.", secondHighlight.get(FIELD_TEXT));
        assertEquals(208.0, secondHighlight.get(FIELD_START));
        assertEquals(275.0, secondHighlight.get(FIELD_END));
    }

    @Test
    public void testProcessOutput_WithNoRelevantSentences() {
        // Mock context with sentences
        when(translatorContext.getAttachment(KEY_SENTENCES)).thenReturn(sentences);
        
        // Mock model output - empty array to indicate no relevant sentences
        NDArray mockOutput = mock(NDArray.class);
        when(mockOutput.getShape()).thenReturn(new Shape(0));
        when(mockOutput.toLongArray()).thenReturn(new long[]{}); // Empty array = no relevant sentences
        
        NDList mockList = new NDList(mockOutput);
        
        // Process output
        Output output = translator.processOutput(translatorContext, mockList);
        
        // Verify results
        assertNotNull(output);
        byte[] bytes = output.getData().getAsBytes();
        ModelTensors tensorOutput = ModelTensors.fromBytes(bytes);
        List<ModelTensor> modelTensorsList = tensorOutput.getMlModelTensors();
        
        // Should have one tensor with the error
        assertEquals(1, modelTensorsList.size());
        ModelTensor errorTensor = modelTensorsList.get(0);
        assertEquals(FIELD_ERROR, errorTensor.getName());
        
        // Get the error from the dataAsMap
        @SuppressWarnings("unchecked")
        Map<String, Object> dataMap = (Map<String, Object>) errorTensor.getDataAsMap();
        assertNotNull(dataMap);
        
        // Should have an error message
        assertNotNull(dataMap.get(FIELD_ERROR));
        assertEquals("No relevant sentences found", dataMap.get(FIELD_ERROR));
        
        // Should have an empty highlights list
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> highlights = (List<Map<String, Object>>) dataMap.get(FIELD_HIGHLIGHTS);
        assertNotNull(highlights);
        assertEquals(0, highlights.size());
    }

    @Test
    public void testProcessOutput_WithEmptyNDList() {
        // Mock context with sentences
        when(translatorContext.getAttachment(KEY_SENTENCES)).thenReturn(sentences);
        
        // Process output with empty NDList
        Output output = translator.processOutput(translatorContext, new NDList());
        
        // Verify results
        assertNotNull(output);
        byte[] bytes = output.getData().getAsBytes();
        ModelTensors tensorOutput = ModelTensors.fromBytes(bytes);
        List<ModelTensor> modelTensorsList = tensorOutput.getMlModelTensors();
        
        // Should have one tensor with the error
        assertEquals(1, modelTensorsList.size());
        ModelTensor errorTensor = modelTensorsList.get(0);
        
        // Get the error from the dataAsMap
        @SuppressWarnings("unchecked")
        Map<String, Object> dataMap = (Map<String, Object>) errorTensor.getDataAsMap();
        assertNotNull(dataMap);
        
        // Should have an error message
        assertNotNull(dataMap.get(FIELD_ERROR));
        
        // Should have an empty highlights list
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> highlights = (List<Map<String, Object>>) dataMap.get(FIELD_HIGHLIGHTS);
        assertNotNull(highlights);
        assertEquals(0, highlights.size());
    }

    @Test
    public void testProcessOutput_WithNoSentences() {
        // Mock context with no sentences
        when(translatorContext.getAttachment(KEY_SENTENCES)).thenReturn(new ArrayList<>());
        
        // Mock model output
        NDArray mockOutput = mock(NDArray.class);
        when(mockOutput.getShape()).thenReturn(new Shape(0));
        when(mockOutput.toLongArray()).thenReturn(new long[0]);
        
        NDList mockList = new NDList(mockOutput);
        
        // Process output
        Output output = translator.processOutput(translatorContext, mockList);
        
        // Verify results
        assertNotNull(output);
        byte[] bytes = output.getData().getAsBytes();
        ModelTensors tensorOutput = ModelTensors.fromBytes(bytes);
        List<ModelTensor> modelTensorsList = tensorOutput.getMlModelTensors();
        
        // Should have one tensor with the error
        assertEquals(1, modelTensorsList.size());
        ModelTensor errorTensor = modelTensorsList.get(0);
        
        // Get the error from the dataAsMap
        @SuppressWarnings("unchecked")
        Map<String, Object> dataMap = (Map<String, Object>) errorTensor.getDataAsMap();
        assertNotNull(dataMap);
        
        // Should have an error message
        assertNotNull(dataMap.get(FIELD_ERROR));
        assertEquals("No sentences found in context", dataMap.get(FIELD_ERROR));
        
        // Should have an empty highlights list
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> highlights = (List<Map<String, Object>>) dataMap.get(FIELD_HIGHLIGHTS);
        assertNotNull(highlights);
        assertEquals(0, highlights.size());
    }

    @Test
    public void testCreateDefault() {
        MLModelConfig modelConfig = mock(MLModelConfig.class);
        SentenceHighlightingQATranslator translator = SentenceHighlightingQATranslator.create(modelConfig);
        assertNotNull(translator);
        assertNotNull(translator.getSegmenter());
        assertEquals(DefaultSentenceSegmenter.class, translator.getSegmenter().getClass());
    }

    /**
     * Test for getChunkEncoding method with overflow encoding handling.
     * This tests the code in lines 235-242 of the SentenceHighlightingQATranslator class.
     */
    @Test
    public void testGetChunkEncoding_WithOverflowEncodings() throws Exception {
        // Create a mock encoding with overflow encodings
        ai.djl.huggingface.tokenizers.Encoding mockEncoding = mock(ai.djl.huggingface.tokenizers.Encoding.class);
        ai.djl.huggingface.tokenizers.Encoding mockOverflowEncoding1 = mock(ai.djl.huggingface.tokenizers.Encoding.class);
        ai.djl.huggingface.tokenizers.Encoding mockOverflowEncoding2 = mock(ai.djl.huggingface.tokenizers.Encoding.class);

        // Set up the mock to return overflow encodings
        when(mockEncoding.getOverflowing())
            .thenReturn(new ai.djl.huggingface.tokenizers.Encoding[] { mockOverflowEncoding1, mockOverflowEncoding2 });

        // Create a mock tokenizer
        ai.djl.huggingface.tokenizers.HuggingFaceTokenizer mockTokenizer = mock(ai.djl.huggingface.tokenizers.HuggingFaceTokenizer.class);
        when(mockTokenizer.encode("test question", "test context")).thenReturn(mockEncoding);

        // Create a translator with the mock tokenizer
        SentenceHighlightingQATranslator translator = SentenceHighlightingQATranslator.builder().build();

        // Set the tokenizer using reflection
        java.lang.reflect.Field tokenizerField = SentenceHighlightingQATranslator.class.getDeclaredField("tokenizer");
        tokenizerField.setAccessible(true);
        tokenizerField.set(translator, mockTokenizer);

        // Access the private getChunkEncoding method using reflection
        java.lang.reflect.Method getChunkEncodingMethod = SentenceHighlightingQATranslator.class
            .getDeclaredMethod("getChunkEncoding", String.class, String.class, int.class);
        getChunkEncodingMethod.setAccessible(true);

        // Test chunk 0 (initial chunk)
        ai.djl.huggingface.tokenizers.Encoding result0 = (ai.djl.huggingface.tokenizers.Encoding) getChunkEncodingMethod
            .invoke(translator, "test question", "test context", 0);
        assertEquals(mockEncoding, result0);

        // Test chunk 1 (first overflow chunk)
        ai.djl.huggingface.tokenizers.Encoding result1 = (ai.djl.huggingface.tokenizers.Encoding) getChunkEncodingMethod
            .invoke(translator, "test question", "test context", 1);
        assertEquals(mockOverflowEncoding1, result1);

        // Test chunk 2 (second overflow chunk)
        ai.djl.huggingface.tokenizers.Encoding result2 = (ai.djl.huggingface.tokenizers.Encoding) getChunkEncodingMethod
            .invoke(translator, "test question", "test context", 2);
        assertEquals(mockOverflowEncoding2, result2);
    }

    /**
     * Test for getChunkEncoding method with invalid chunk number.
     * This tests the exception branch in lines 240-241 of the SentenceHighlightingQATranslator class.
     */
    @Test
    public void testGetChunkEncoding_WithInvalidChunkNumber() throws Exception {
        // Create a mock encoding with some overflow encodings
        ai.djl.huggingface.tokenizers.Encoding mockEncoding = mock(ai.djl.huggingface.tokenizers.Encoding.class);
        ai.djl.huggingface.tokenizers.Encoding mockOverflowEncoding = mock(ai.djl.huggingface.tokenizers.Encoding.class);

        // Set up the mock to return a single overflow encoding
        when(mockEncoding.getOverflowing()).thenReturn(new ai.djl.huggingface.tokenizers.Encoding[] { mockOverflowEncoding });

        // Create a mock tokenizer
        ai.djl.huggingface.tokenizers.HuggingFaceTokenizer mockTokenizer = mock(ai.djl.huggingface.tokenizers.HuggingFaceTokenizer.class);
        when(mockTokenizer.encode("test question", "test context")).thenReturn(mockEncoding);

        // Create a translator with the mock tokenizer
        SentenceHighlightingQATranslator translator = SentenceHighlightingQATranslator.builder().build();

        // Set the tokenizer using reflection
        java.lang.reflect.Field tokenizerField = SentenceHighlightingQATranslator.class.getDeclaredField("tokenizer");
        tokenizerField.setAccessible(true);
        tokenizerField.set(translator, mockTokenizer);

        // Access the private getChunkEncoding method using reflection
        java.lang.reflect.Method getChunkEncodingMethod = SentenceHighlightingQATranslator.class
            .getDeclaredMethod("getChunkEncoding", String.class, String.class, int.class);
        getChunkEncodingMethod.setAccessible(true);

        // Test with invalid chunk number (larger than available overflow encodings)
        try {
            getChunkEncodingMethod.invoke(translator, "test question", "test context", 2);
            fail("Should have thrown an exception for invalid chunk number");
        } catch (java.lang.reflect.InvocationTargetException e) {
            // Verify that the cause is an IllegalArgumentException
            assertTrue(e.getCause() instanceof IllegalArgumentException);
            assertEquals("Invalid chunk number: 2", e.getCause().getMessage());
        }
    }

    /**
     * Test for getChunkEncoding method with null overflow encodings.
     * This tests another branch in the handling of overflow encodings in the getChunkEncoding method.
     */
    @Test
    public void testGetChunkEncoding_WithNullOverflowEncodings() throws Exception {
        // Create a mock encoding with null overflow encodings
        ai.djl.huggingface.tokenizers.Encoding mockEncoding = mock(ai.djl.huggingface.tokenizers.Encoding.class);

        // Set up the mock to return null for overflow encodings
        when(mockEncoding.getOverflowing()).thenReturn(null);

        // Create a mock tokenizer
        ai.djl.huggingface.tokenizers.HuggingFaceTokenizer mockTokenizer = mock(ai.djl.huggingface.tokenizers.HuggingFaceTokenizer.class);
        when(mockTokenizer.encode("test question", "test context")).thenReturn(mockEncoding);

        // Create a translator with the mock tokenizer
        SentenceHighlightingQATranslator translator = SentenceHighlightingQATranslator.builder().build();

        // Set the tokenizer using reflection
        java.lang.reflect.Field tokenizerField = SentenceHighlightingQATranslator.class.getDeclaredField("tokenizer");
        tokenizerField.setAccessible(true);
        tokenizerField.set(translator, mockTokenizer);

        // Access the private getChunkEncoding method using reflection
        java.lang.reflect.Method getChunkEncodingMethod = SentenceHighlightingQATranslator.class
            .getDeclaredMethod("getChunkEncoding", String.class, String.class, int.class);
        getChunkEncodingMethod.setAccessible(true);

        // Test with chunk number > 0 when overflow encodings are null
        try {
            getChunkEncodingMethod.invoke(translator, "test question", "test context", 1);
            fail("Should have thrown an exception for invalid chunk number with null overflow encodings");
        } catch (java.lang.reflect.InvocationTargetException e) {
            // Verify that the cause is an IllegalArgumentException
            assertTrue(e.getCause() instanceof IllegalArgumentException);
            assertEquals("Invalid chunk number: 1", e.getCause().getMessage());
        }
    }
}
