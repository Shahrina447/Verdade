from fastapi import APIRouter, HTTPException

from app.schemas.history import HistoryItem
from app.services import history as history_svc

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=list[HistoryItem])
def get_history() -> list[HistoryItem]:
    return history_svc.load()


@router.delete("/{item_id}")
def delete_item(item_id: str):
    hist, found = history_svc.delete_one(history_svc.load(), item_id)
    if not found:
        raise HTTPException(status_code=404, detail="Item not found.")
    history_svc.save(hist)
    return {"deleted": item_id}


@router.delete("")
def clear_history():
    history_svc.save([])
    return {"cleared": True}
