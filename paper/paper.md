# The Visual Instruction Following Evaluation Benchmark (VIF-Eval): A Comprehensive Framework for Evaluating Compositional and Editorial Image Generation

**Authors:** [Your Name/Team]  
**Date:** November 2025  
**Institution:** [Your Institution]

---

## Abstract

This paper introduces the Visual Instruction Following Evaluation Benchmark (VIF-Eval), a comprehensive framework for assessing the capabilities of generative visual models in real-world scenarios involving complex image creation, editing, and compositional reasoning. Unlike traditional text-to-image benchmarks that focus on simple prompt adherence, VIF-Eval evaluates models across multiple dimensions including complex compositional reasoning, text rendering accuracy, constraint satisfaction problems (CSP), character consistency, and sketch-to-render translation. We evaluate three state-of-the-art models—GPT Image 1, DALL-E 3, and Gemini 2.5 Flash Image (Nano Banana)—across 47 diverse prompts with 167+ constraints. Our results reveal significant performance variations across constraint types, with models achieving 54.7-58.1% overall pass rates on standard constraints but struggling with text rendering (13.0% pass rate) and complex compositional tasks. The benchmark provides actionable insights for improving generative AI systems and establishes a foundation for future research in visual instruction following.

**Keywords:** Image Generation, Visual AI, Benchmark Evaluation, Compositional Reasoning, Text-to-Image Models

---

## 1. Introduction

### 1.1 Background

Generative visual models have achieved remarkable progress in recent years, enabling high-quality image synthesis from text prompts. Models like DALL-E 3, Midjourney, Stable Diffusion, and GPT Image 1 have demonstrated impressive capabilities in creating photorealistic and artistic images. However, as these models are increasingly deployed in real-world applications—including e-commerce, advertising, design, and creative industries—there is a growing need to evaluate their performance beyond simple aesthetic quality.

### 1.2 Motivation and Industry Relevance

Real-world visual generation tasks often require:
- **Complex Compositional Reasoning**: Correctly placing multiple objects with specific attributes and spatial relationships
- **Accurate Text Rendering**: Generating legible text within images for posters, labels, and advertisements
- **Constraint Satisfaction**: Following mathematical and logical constraints (e.g., nutrition labels, data tables)
- **Consistency**: Maintaining character appearance and style across multiple scenes
- **Precision Editing**: Modifying specific image regions while preserving context

Current benchmarks primarily focus on prompt-image alignment using metrics like CLIP score or human preference ratings, but fail to assess these critical capabilities systematically.

### 1.3 Gap Analysis

Existing text-to-image evaluation frameworks have several limitations:

1. **Limited Task Diversity**: Most benchmarks focus on single-object generation or simple scene composition
2. **Lack of Automated Precision Metrics**: Heavy reliance on human evaluation makes large-scale assessment impractical
3. **Missing Constraint Validation**: No systematic evaluation of logical, mathematical, or compositional constraints
4. **Insufficient Real-World Scenarios**: Benchmarks often use abstract or artistic prompts rather than practical use cases

VIF-Eval addresses these gaps by providing:
- A comprehensive task taxonomy covering 9 constraint types
- Automated evaluation metrics for each constraint type
- Real-world inspired prompts (nutrition labels, posters, product compositions)
- Systematic error analysis and failure pattern identification

### 1.4 Contributions

This paper makes the following contributions:

1. **Comprehensive Benchmark**: A new evaluation framework with 47 diverse prompts and 167+ constraints
2. **Multi-Model Evaluation**: Systematic comparison of three state-of-the-art models
3. **Automated Evaluation Pipeline**: CLIP-based, OCR-based, and computer vision metrics for objective assessment
4. **Error Analysis**: Detailed categorization of failure modes and model limitations
5. **Open-Source Implementation**: Complete evaluation codebase and dataset for reproducibility

---

## 2. Methodology

### 2.1 Visual Task Taxonomy

VIF-Eval defines a comprehensive taxonomy of visual instruction following tasks:

#### 2.1.1 Complex Prompt Adherence
- **Count Constraints**: Verify correct number of objects (e.g., "three blue mugs")
- **Attribute Constraints**: Verify object attributes (e.g., color, size, material)
- **State Constraints**: Verify object states (e.g., "tipped over", "empty")
- **Spatial Constraints**: Verify spatial relationships (e.g., "left of", "above")

#### 2.1.2 Text Rendering
- **Poster Text**: Rendering legible text in poster designs
- **Label Text**: Accurate text in nutrition labels and product information
- **Banner Text**: Text in advertising banners and headers

#### 2.1.3 Constraint Satisfaction Problems (CSP)
- **Numeric Relations**: Mathematical relationships (A < B, A + B = C)
- **Sorting**: Ordered sequences (ascending, descending)
- **Uniqueness**: All-different constraints
- **Range Constraints**: Value bounds (min ≤ x ≤ max)
- **Complex Operations**: Products, ratios, differences, modulo

#### 2.1.4 Style & Character Consistency
- **Character Consistency**: Same character across multiple scenes
- **Attribute Preservation**: Maintaining clothing, appearance, accessories

#### 2.1.5 Image-to-Image Translation
- **Sketch-to-Render**: Converting simple sketches to photorealistic renders
- **Structural Fidelity**: Preserving structural elements while enhancing detail

### 2.2 Dataset Creation

#### 2.2.1 Prompt Sourcing

Our prompt suite consists of 47 prompts across 5 categories:

1. **Composition (6 prompts)**: Multi-object scenes with count, attribute, state, and spatial constraints
2. **Poster Text (6 prompts)**: Advertising posters requiring accurate text rendering
3. **Nutrition Labels (6 prompts)**: Food labels with structured data and text
4. **CSP Demo (13 prompts)**: Mathematical and logical constraint satisfaction
5. **Advanced (6 prompts)**: Complex multi-constraint scenarios
6. **Character Consistency (6 prompts)**: Character appearance across scenes
7. **Sketch-to-Render (5 prompts)**: Sketch translation tasks

Prompts were designed to:
- Reflect real-world use cases (e-commerce, advertising, design)
- Include multiple constraints per prompt for comprehensive evaluation
- Cover edge cases and challenging scenarios
- Balance difficulty across constraint types

#### 2.2.2 Data Annotation

Each prompt is annotated with:
- **Category**: Task category (composition, text, CSP, etc.)
- **Constraints**: List of constraint objects with:
  - Constraint type (count, attribute, text, CSP, etc.)
  - Required fields (object, target, field_map, etc.)
  - Expected values or relationships
- **Metadata**: Prompt ID, description, difficulty level

#### 2.2.3 Dataset Statistics

- **Total Prompts**: 47
- **Total Constraints**: 167-172 (varies by model due to special evaluators)
- **Constraint Types**: 11 distinct types
- **Categories**: 7 categories
- **Average Constraints per Prompt**: 3.5

### 2.3 Evaluation Metrics

#### 2.3.1 Automated Metrics

**CLIP-Based Metrics:**
- **Composition Scoring**: CLIP similarity for count, attribute, and state constraints
- **Negative Constraint Scoring**: Inverse CLIP similarity for forbidden concepts
- **Prompt Adherence**: CLIP score for overall prompt-image alignment

**OCR-Based Metrics:**
- **Text Accuracy**: Character Error Rate (CER) for text rendering
- **Label Parsing**: Field extraction accuracy for structured data
- **CSP Validation**: Numeric value extraction and constraint satisfaction

**Computer Vision Metrics:**
- **SSIM**: Structural Similarity Index for sketch-to-render tasks
- **Edge Alignment**: Intersection over Union (IoU) of edge maps
- **Object Detection**: GroundingDINO for spatial relationship validation
- **Face Recognition**: InsightFace for character consistency

**Scoring Methodology:**
- All scores normalized to [0, 1] range
- Pass threshold: score > 0.5
- Binary constraints: 1.0 (satisfied) or 0.0 (not satisfied)
- Continuous constraints: Logistic or exponential decay functions

#### 2.3.2 Constraint-Specific Metrics

1. **Count**: Margin-based logistic scoring comparing target count to alternatives
2. **Attribute**: CLIP similarity margin between attribute and plain object
3. **Text**: Character Error Rate with exponential decay: exp(-3.0 × CER)
4. **CSP**: Binary satisfaction or relative error-based scoring
5. **Spatial**: Binary validation of bounding box relationships
6. **Character Consistency**: Average of face similarity, detection consistency, and attribute consistency

#### 2.3.3 Aggregation Metrics

- **Pass Rate**: Percentage of constraints with score > 0.5
- **Average Score**: Mean score across all constraints
- **Per-Type Performance**: Pass rates and average scores by constraint type
- **Per-Category Performance**: Pass rates and average scores by prompt category

### 2.4 Models Evaluated

#### 2.4.1 GPT Image 1
- **Provider**: OpenAI
- **Model**: gpt-image-1
- **Configuration**: 1024×1024, high quality
- **API**: OpenAI Images API

#### 2.4.2 DALL-E 3
- **Provider**: OpenAI
- **Model**: dall-e-3
- **Configuration**: 1024×1024, HD quality
- **API**: OpenAI Images API
- **Note**: Evaluation incomplete due to API billing limits (11/47 prompts completed)

#### 2.4.3 Gemini 2.5 Flash Image (Nano Banana)
- **Provider**: Google (via OpenRouter)
- **Model**: google/gemini-2.5-flash-image
- **Configuration**: 1024×1024, high quality
- **API**: OpenRouter API (OpenAI-compatible interface)

---

## 3. Results

### 3.1 Overall Performance

| Model | Prompts | Constraints | Passed | Pass Rate | Errors |
|-------|---------|-------------|--------|-----------|--------|
| GPT Image 1 | 47 | 167 | 97 | 58.1% | 1 |
| Nano Banana | 47 | 172 | 94 | 54.7% | 0 |
| DALL-E 3 | 47 | 76 | 13 | 17.1% | 36 |

**Key Findings:**
- GPT Image 1 achieves the highest overall pass rate (58.1%)
- Nano Banana performs comparably (54.7%), showing strong competition
- DALL-E 3 evaluation incomplete; partial results show 17.1% pass rate (likely due to incomplete evaluation)

### 3.2 Performance by Constraint Type

#### 3.2.1 GPT Image 1

| Constraint Type | Total | Passed | Pass Rate | Avg Score |
|----------------|-------|--------|-----------|-----------|
| Spatial | 2 | 2 | 100.0% | 1.000 |
| Character Consistency | 4 | 4 | 100.0% | 0.987 |
| CSP | 20 | 20 | 100.0% | 1.000 |
| Negative | 23 | 23 | 100.0% | 0.940 |
| State | 6 | 5 | 83.3% | 0.556 |
| Attribute | 20 | 11 | 55.0% | 0.521 |
| Table Slot | 42 | 20 | 47.6% | 0.488 |
| Count | 16 | 5 | 31.2% | 0.491 |
| Logic | 6 | 3 | 50.0% | 0.387 |
| Text | 23 | 3 | 13.0% | 0.164 |
| Sketch-to-Render | 5 | 1 | 20.0% | 0.456 |

#### 3.2.2 Nano Banana

| Constraint Type | Total | Passed | Pass Rate | Avg Score |
|----------------|-------|--------|-----------|-----------|
| Spatial | 2 | 2 | 100.0% | 1.000 |
| Character Consistency | 4 | 4 | 100.0% | 0.979 |
| Negative | 24 | 24 | 100.0% | 0.947 |
| CSP | 20 | 19 | 95.0% | 0.950 |
| Logic | 6 | 4 | 66.7% | 0.504 |
| Table Slot | 42 | 23 | 54.8% | 0.551 |
| Attribute | 21 | 9 | 42.9% | 0.517 |
| State | 6 | 2 | 33.3% | 0.537 |
| Count | 17 | 3 | 17.6% | 0.476 |
| Sketch-to-Render | 5 | 1 | 20.0% | 0.451 |
| Text | 25 | 3 | 12.0% | 0.149 |

### 3.3 Performance by Category

#### GPT Image 1
- **CSP Demo**: 100.0% pass rate (20/20 constraints)
- **Composition**: 77.8% pass rate (21/27 constraints)
- **Advanced**: 50.0% pass rate (14/28 constraints)
- **Nutrition Label**: 50.0% pass rate (30/60 constraints)
- **Poster Text**: 30.4% pass rate (7/23 constraints)

#### Nano Banana
- **CSP Demo**: 95.0% pass rate (19/20 constraints)
- **Composition**: 70.4% pass rate (19/27 constraints)
- **Advanced**: 50.0% pass rate (14/28 constraints)
- **Nutrition Label**: 50.0% pass rate (30/60 constraints)
- **Poster Text**: 30.4% pass rate (7/23 constraints)

### 3.4 Key Insights

1. **Strong Performance Areas:**
   - Spatial relationships: 100% pass rate (both models)
   - Character consistency: 100% pass rate (both models)
   - CSP constraints: 95-100% pass rate
   - Negative constraints: 100% pass rate (avoiding forbidden concepts)

2. **Challenging Areas:**
   - Text rendering: 12-13% pass rate (critical weakness)
   - Count constraints: 17.6-31.2% pass rate
   - Sketch-to-render: 20% pass rate

3. **Model Comparison:**
   - GPT Image 1 slightly outperforms Nano Banana overall (58.1% vs 54.7%)
   - Nano Banana performs better on logic constraints (66.7% vs 50.0%)
   - Both models struggle similarly with text rendering and counting

---

## 4. Model Behavior Analysis

### 4.1 Error Patterns

#### 4.1.1 Text Rendering Failures

**Common Issues:**
- Garbled or unreadable text
- Missing text entirely
- Incorrect characters (OCR confusion)
- Text in wrong location

**Impact:** 87-88% failure rate across all models

**Example:** Poster text prompts often result in decorative patterns instead of legible text.

#### 4.1.2 Count Constraint Failures

**Common Issues:**
- Incorrect object counts (e.g., generating 2 objects when 3 requested)
- Objects partially occluded, making counting difficult
- CLIP-based counting struggles with similar objects

**Impact:** 68.8-82.4% failure rate

**Root Cause:** CLIP embeddings may not capture precise object counts, especially for similar objects.

#### 4.1.3 Composition Failures

**Common Issues:**
- Attribute mismatches (wrong colors, sizes)
- Spatial relationship errors
- Missing objects or attributes

**Impact:** Varies by constraint type (attribute: 45-57% failure, spatial: 0% failure)

### 4.2 Constraint Satisfaction Analysis

#### 4.2.1 CSP Performance

Both models excel at CSP constraints (95-100% pass rate), indicating strong:
- OCR accuracy (DeepSeek-OCR integration)
- Numeric parsing capabilities
- Mathematical constraint validation

#### 4.2.2 Character Consistency

100% pass rate demonstrates:
- Effective face recognition (InsightFace)
- Robust attribute preservation
- Strong cross-scene consistency

### 4.3 Failure Mode Categorization

1. **Text Generation Limitations**: Models struggle to generate legible text
2. **Counting Precision**: CLIP-based counting is imprecise for similar objects
3. **Complex Composition**: Multi-object scenes with multiple constraints are challenging
4. **Sketch Fidelity**: Independent sketch/render generation leads to structural mismatches

---

## 5. Case Studies

### 5.1 Case Study 1: Text Rendering (text_005)

**Prompt:** "A poster with the text 'SUMMER SALE' in large bold letters"

**Results:**
- GPT Image 1: Failed (text not legible)
- Nano Banana: Failed (text garbled)
- DALL-E 3: Not evaluated

**Analysis:** Both models generated decorative text-like patterns but failed to produce legible characters. This highlights a fundamental limitation in current text-to-image models for text rendering tasks.

### 5.2 Case Study 2: Complex Composition (comp_001)

**Prompt:** "A photo of three blue mugs and two red plates on a wooden table. One of the blue mugs is tipped over. The largest blue mug is on the left side of the table, and the smallest red plate is on the right side."

**Constraints:**
- Count: 3 mugs, 2 plates
- Attribute: Blue mugs, red plates
- State: One mug tipped over
- Spatial: Largest mug left, smallest plate right

**Results:**
- GPT Image 1: 2/6 constraints passed (spatial relationships correct)
- Nano Banana: 2/6 constraints passed
- DALL-E 3: Partial evaluation

**Analysis:** Models successfully handle spatial relationships but struggle with precise counting and attribute verification.

### 5.3 Case Study 3: CSP Constraint (csp_01_numbers_row)

**Prompt:** CSP task with numeric relationships

**Results:**
- GPT Image 1: Passed (100%)
- Nano Banana: Passed (95%)
- DALL-E 3: Not evaluated

**Analysis:** Strong performance on mathematical constraints demonstrates effective OCR and numeric parsing capabilities.

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Incomplete DALL-E 3 Evaluation**: API billing limits prevented full evaluation
2. **Limited Human Evaluation**: Primarily automated metrics; human preference scores not included
3. **Cost/Latency Metrics**: Not systematically tracked in current evaluation
4. **In-Painting Evaluation**: Not included in current benchmark (future addition)

### 6.2 Future Directions

1. **Expand Dataset**: Add more prompts, especially for in-painting and out-painting tasks
2. **Human Evaluation**: Incorporate human preference scores and compositional accuracy ratings
3. **Cost Analysis**: Systematic tracking of API costs and generation latency
4. **Tool Usage Analysis**: Evaluate model selection of generation vs. editing tools
5. **Trajectory Visualization**: Analyze multi-step generation processes
6. **Additional Models**: Evaluate Midjourney, Stable Diffusion 3, and other models

---

## 8. Conclusion

The Visual Instruction Following Evaluation Benchmark (VIF-Eval) provides a comprehensive framework for assessing generative visual models beyond simple prompt adherence. Our evaluation of three state-of-the-art models reveals:

1. **Strengths**: Models excel at spatial relationships, character consistency, and constraint satisfaction problems
2. **Weaknesses**: Critical limitations in text rendering (12-13% pass rate) and precise counting (17-31% pass rate)
3. **Comparability**: GPT Image 1 and Nano Banana show similar overall performance (54-58% pass rates)

The benchmark establishes a foundation for systematic evaluation of visual instruction following capabilities and provides actionable insights for model improvement. Future work should expand the dataset, incorporate human evaluation, and track cost/latency metrics to provide a complete assessment framework.

---

## 9. References

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
2. Ramesh, A., et al. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents. NeurIPS.
3. Betker, J., et al. (2023). Improving Image Generation with Better Captions. OpenAI Blog.
4. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
5. DeepSeek-OCR: https://github.com/deepseek-ai/DeepSeek-OCR
6. GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
7. InsightFace: https://github.com/deepinsight/insightface

---

## 10. Appendix

### 9.1 Evaluation Logic

See `evaluation_logic.txt` for detailed documentation of each evaluator's methodology.

### 9.2 Dataset Statistics

- Total Prompts: 47
- Total Constraints: 167-172
- Constraint Types: 11
- Categories: 7

### 9.3 Additional Visualizations

See `paper_assets/figures/` for:
- Pass rate comparisons by constraint type
- Average score comparisons
- Overall performance visualizations

### 9.4 Case Study Images

See `paper_assets/case_studies/` for side-by-side model comparisons.

### 9.5 Code and Data Availability

The complete evaluation codebase, prompts, and results are available at: [Repository URL]

---

**End of Paper**

