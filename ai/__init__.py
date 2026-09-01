"""TRINETRA AI pipeline.

Owns the boundary between messy reality and clean data: five media adapters
collapse into one FrameEnvelope, the CV pipeline emits one EventEnvelope.

No business layer may know whether the source is live, recorded or synthetic.
"""

__all__ = ["PIPELINE_VERSION"]

PIPELINE_VERSION = "0.1.0"
