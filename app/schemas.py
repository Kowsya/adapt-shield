"""
app/schemas.py
==============
Pydantic schemas used by the FastAPI app.
Defines request/response data shapes with validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict


class TrafficFeatures(BaseModel):
    """
    Input schema: one network flow record.
    All 77 features are optional (missing = 0).
    The API name uses underscores; they map to dataset column names.
    """
    Protocol: float = Field(default=0, description="Protocol (6=TCP, 17=UDP)")
    Flow_Duration: float = Field(default=0, description="Duration of the flow in microseconds")
    Total_Fwd_Packets: float = Field(default=0, description="Total packets in forward direction")
    Total_Backward_Packets: float = Field(default=0, description="Total packets in backward direction")
    Fwd_Packets_Length_Total: float = Field(default=0)
    Bwd_Packets_Length_Total: float = Field(default=0)
    Fwd_Packet_Length_Max: float = Field(default=0)
    Fwd_Packet_Length_Min: float = Field(default=0)
    Fwd_Packet_Length_Mean: float = Field(default=0)
    Bwd_Packet_Length_Max: float = Field(default=0)
    Bwd_Packet_Length_Min: float = Field(default=0)
    Bwd_Packet_Length_Mean: float = Field(default=0)
    Flow_Bytes_s: float = Field(default=0, description="Flow bytes per second")
    Flow_Packets_s: float = Field(default=0, description="Flow packets per second")
    Flow_IAT_Mean: float = Field(default=0)
    Flow_IAT_Std: float = Field(default=0)
    Flow_IAT_Max: float = Field(default=0)
    Flow_IAT_Min: float = Field(default=0)
    Fwd_IAT_Total: float = Field(default=0)
    Fwd_IAT_Mean: float = Field(default=0)
    Fwd_IAT_Std: float = Field(default=0)
    Fwd_IAT_Max: float = Field(default=0)
    Fwd_IAT_Min: float = Field(default=0)
    Bwd_IAT_Total: float = Field(default=0)
    Bwd_IAT_Mean: float = Field(default=0)
    Bwd_IAT_Std: float = Field(default=0)
    Bwd_IAT_Max: float = Field(default=0)
    Bwd_IAT_Min: float = Field(default=0)
    Fwd_Header_Length: float = Field(default=0)
    Bwd_Header_Length: float = Field(default=0)
    Fwd_Packets_s: float = Field(default=0)
    Bwd_Packets_s: float = Field(default=0)
    Packet_Length_Min: float = Field(default=0)
    Packet_Length_Max: float = Field(default=0)
    Packet_Length_Mean: float = Field(default=0)
    Packet_Length_Std: float = Field(default=0)
    Packet_Length_Variance: float = Field(default=0)
    FIN_Flag_Count: float = Field(default=0)
    SYN_Flag_Count: float = Field(default=0, description="High = possible DDoS/SYN flood")
    RST_Flag_Count: float = Field(default=0)
    PSH_Flag_Count: float = Field(default=0)
    ACK_Flag_Count: float = Field(default=0)
    URG_Flag_Count: float = Field(default=0)
    ECE_Flag_Count: float = Field(default=0)
    Down_Up_Ratio: float = Field(default=0)
    Avg_Packet_Size: float = Field(default=0)
    Avg_Fwd_Segment_Size: float = Field(default=0)
    Avg_Bwd_Segment_Size: float = Field(default=0)
    Subflow_Fwd_Packets: float = Field(default=0)
    Subflow_Fwd_Bytes: float = Field(default=0)
    Subflow_Bwd_Packets: float = Field(default=0)
    Subflow_Bwd_Bytes: float = Field(default=0)
    Init_Fwd_Win_Bytes: float = Field(default=0)
    Init_Bwd_Win_Bytes: float = Field(default=0)
    Fwd_Act_Data_Packets: float = Field(default=0)
    Fwd_Seg_Size_Min: float = Field(default=0)
    Active_Mean: float = Field(default=0)
    Active_Std: float = Field(default=0)
    Active_Max: float = Field(default=0)
    Active_Min: float = Field(default=0)
    Idle_Mean: float = Field(default=0)
    Idle_Std: float = Field(default=0)
    Idle_Max: float = Field(default=0)
    Idle_Min: float = Field(default=0)

    class Config:
        extra = "allow"   # Allow all 77 features to be passed


class PredictionResponse(BaseModel):
    """What the /predict endpoint returns."""
    prediction: str = Field(description="Attack class: BENIGN, DDoS, DoS, etc.")
    is_malicious: bool = Field(description="True if attack detected")
    confidence: float = Field(description="Model confidence 0-1")
    action: str = Field(description="ALLOW or BLOCK")
    processing_time_ms: float = Field(description="Inference time in ms")
    class_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Probability for each class"
    )


class BatchRequest(BaseModel):
    """Input for batch prediction."""
    records: list[TrafficFeatures]


class BatchResponse(BaseModel):
    """Output for batch prediction."""
    predictions: list[PredictionResponse]
    total: int
    blocked_count: int
    allowed_count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    blocked_requests: int
    allowed_requests: int
    block_rate_percent: float