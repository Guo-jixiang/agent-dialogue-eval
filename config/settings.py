from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Evaluator LLM
    EVAL_LLM_API_KEY: str = ""
    EVAL_LLM_API_KEY_2: str = ""
    EVAL_LLM_API_KEY_3: str = ""
    EVAL_LLM_BASE_URL: str = "https://api.openai.com/v1"
    EVAL_LLM_MODEL: str = "gpt-4o"

    # Simulator LLM
    SIM_LLM_API_KEY: str = ""
    SIM_LLM_BASE_URL: str = "https://api.openai.com/v1"
    SIM_LLM_MODEL: str = "gpt-4o-mini"

    # Agent under test (LLM-simulated)
    AGENT_LLM_API_KEY: str = ""
    AGENT_LLM_BASE_URL: str = "https://api.openai.com/v1"
    AGENT_LLM_MODEL: str = ""

    # Friday digital-human API (external LLM call)
    FRIDAY_API_URL: str = ""
    FRIDAY_API_KEY: str = ""
    FRIDAY_LLM_MODEL: str = ""  # model name sent to Friday API, if applicable

    # AI Assistant LLM (isolated from eval/sim keys — leave empty to fallback to EVAL_LLM)
    # Recommended: use a lower-cost model (e.g. gpt-4o-mini) for the assistant
    ASSISTANT_LLM_API_KEY: str = ""
    ASSISTANT_LLM_BASE_URL: str = ""
    ASSISTANT_LLM_MODEL: str = ""

    # Evaluation parameters
    MAX_TURNS: int = 30

    # Simulation parameters
    PERSONA_VARIANT_COUNT: int = 14
    SIM_CONCURRENCY: int = 72
    EVAL_CONCURRENCY: int = 72
    LLM_MAX_CONCURRENCY: int = 50  # global cap on concurrent LLM API calls
    EVAL_BATCH_SIZE: int = 72

    # LLM rate limiting & retry
    LLM_MAX_RETRIES: int = 5
    LLM_RATE_LIMIT_RPS: float = 0.0  # 0 = disabled; set to e.g. 5.0 to cap requests/sec
    LLM_RATE_LIMIT_BURST: int = 10
    LLM_JSON_MAX_TOKENS: int = 32768  # default max_tokens for chat_json() calls

    # Self-consistency & confidence
    SELF_CONSISTENCY_RUNS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.6
    DIVERGENCE_THRESHOLD: float = 0.2

    # Anomaly detection thresholds
    ANOMALY_FULL_SCORE_RATIO: float = 0.9
    ANOMALY_ZERO_SCORE_RATIO: float = 0.7

    # Evidence validation
    EVIDENCE_VALIDATION_ENABLED: bool = True

    # Dimension weights
    WEIGHT_FLOW: float = 0.25
    WEIGHT_CONSTRAINT: float = 0.20
    WEIGHT_KNOWLEDGE: float = 0.20
    WEIGHT_STYLE: float = 0.10
    WEIGHT_COHERENCE: float = 0.10
    WEIGHT_SAFETY: float = 0.10
    WEIGHT_ADAPTABILITY: float = 0.05

    # Paths
    PROMPTS_DIR: Path = Path(__file__).parent / "prompts"


settings = Settings()
