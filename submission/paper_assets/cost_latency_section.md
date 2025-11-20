## 6. Cost and Latency Analysis

### 6.1 API Costs

Estimated costs for the evaluation (based on current API pricing):

| Model | Images Generated | Cost per Image | Total Cost |
|-------|------------------|----------------|------------|
| GPT Image 1 | 47 | $0.040 | $1.88 |
| Nano Banana | 47 | $0.005 | $0.24 |
| DALL-E 3 | 11 | $0.040 | $0.44 |

**Note**: Costs are estimates based on published API pricing. Actual costs may vary.

### 6.2 Latency

Latency measurements were not systematically tracked in this evaluation. Future work should include:

- Time-to-first-image (TTFI) measurements
- End-to-end generation time
- API response time analysis
- Comparison of synchronous vs. asynchronous generation

### 6.3 Cost-Performance Analysis

Based on estimated costs and performance:

- **Most Cost-Effective**: Nano Banana (~$0.24 for 47 images, 54.7% pass rate)
- **Best Performance**: GPT Image 1 (~$1.88 for 47 images, 58.1% pass rate)
- **Cost per Passed Constraint**:
  - GPT Image 1: ~$0.019 per passed constraint
  - Nano Banana: ~$0.003 per passed constraint

