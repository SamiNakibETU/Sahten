"""
API Dependencies and Utilities
"""

from typing import Annotated
from fastapi import Depends, Header, HTTPException, status


async def verify_content_type(content_type: Annotated[str | None, Header()] = None):
    """Verify request content type for POST requests"""
    if content_type and not content_type.startswith("application/json"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/json",
        )


# Placeholder for future dependencies
# - get_rag_pipeline()
# - get_data_loaders()
# - rate_limiting
# - authentication (if needed)
