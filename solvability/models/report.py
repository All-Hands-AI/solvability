from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from solvability.llm import completion
from solvability.models.config import LLMConfig
from solvability.models.importance_strategy import ImportanceStrategy


class SolvabilityReport(BaseModel):
    """
    Report for the solvability classifier.
    """

    identifier: str
    """
    The identifier of the solvability model used to generate the report.
    """

    issue: str
    """
    The issue description for which the solvability is predicted.

    This field is exactly the input to the solvability model.
    """

    score: float
    """
    [0, 1]-valued score indicating the likelihood of the issue being solvable.
    """

    cost: float
    """
    Total cost of API calls made to generate the features.
    """

    response_latency: float
    """
    Total response latency of API calls made to generate the features.
    """

    features: dict[str, float]
    """
    [0, 1]-valued scores for each feature in the model.

    These are the values fed to the random forest classifier to generate the solvability score.
    """

    samples: int
    """
    Number of samples used to compute the feature embedding coefficients.
    """

    importance_strategy: ImportanceStrategy
    """
    Strategy used to calculate feature importances.
    """

    feature_importances: dict[str, float]
    """
    Importance scores for each feature in the model.

    Interpretation of these scores depends on the importance strategy used.
    """

    created_at: datetime = Field(default_factory=datetime.now)
    """
    Datetime when the report was created.
    """

    random_state: int | None = None
    """
    Classifier random state used when generating this report.
    """

    metadata: dict[str, Any] | None = None
    """
    Metadata for logging and debugging purposes.
    """

    def summarize(self, llm_config: LLMConfig | None = None) -> str:  # noqa: E501
        """
        Generate a summary of the report using the LLM.
        """
        prompt = f"""
Convert this SolvabilityReport into a concise executive summary for technical stakeholders:

Report Data: {self.model_dump()}

Feature importances are computed using SHAP values, indicating the contribution of each feature to the model's
predictions. The features are ranked by their SHAP values, with higher values indicating greater importance.

Format the output as:
- Lead with the solvability assessment and confidence level (LOW, MEDIUM, HIGH)
- Highlight key contributing factors from feature analysis
- End with actionable insights or recommendations from the contributing factors

Do not include any mention of costs or other performance metrics from the report itself.

Include technical information suitable for software engineers and data scientists. DO NOT include direct mentions of
feature importance values.

Keep the length short, and include recommendations as a simple list.
"""

        response = completion(messages=[{"role": "user", "content": prompt}], llm_config=llm_config)
        return response.choices[0].message.content
