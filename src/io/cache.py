import pickle
from datetime import datetime
from pathlib import Path

from src.io.paths import CACHE_PATH, ensure_dirs
from src.core.plant_context import PlantContext

def _build_snapshot_filename(file_type, data_type, stage, timestamp=None):
    """
    Build standardized snapshot filename.
    
    Format: {file_type}_{data_type}_snapshot_{stage}_{YYYYMMDD_HHMM}.pkl
    
    Parameters
    ----------
    file_type : str
        'events' or 'plants'
    data_type : str
        'exp', 'sim', 'xtrm', 'motor'
    stage : str
        'build', 'sine_fit', etc.
    timestamp : str, optional
        YYYYMMDD_HHMM format. If None, uses current time.
    
    Returns
    -------
    str
        Filename
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{file_type}_{data_type}_snapshot_{stage}_{timestamp}.pkl"


def _get_latest_snapshot(file_type, data_type, stage=None):
    """
    Find latest snapshot file by modification time.
    
    Parameters
    ----------
    file_type : str
        'events' or 'plants'
    data_type : str
        'exp', 'sim', 'xtrm', 'motor'
    stage : str, optional
        If provided, filter by stage. If None, match any stage.
    
    Returns
    -------
    Path
        Path to latest matching file
    """
    if stage:
        pattern = f"{file_type}_{data_type}_snapshot_{stage}_*.pkl"
    else:
        pattern = f"{file_type}_{data_type}_snapshot_*_*.pkl"
    
    matches = list(CACHE_PATH.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No snapshot found matching: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def save_snapshot(obj, file_type, data_type, stage, plants=None):
    """
    Save a snapshot with standardized naming.
    
    Parameters
    ----------
    obj : list or object
        Object(s) to save (events, plants, etc.)
    file_type : str
        'events' or 'plants'
    data_type : str
        'exp', 'sim', 'xtrm', 'motor'
    stage : str
        'build', 'sine_fit', etc.
    plants : list, optional
        If file_type is 'events', optionally attach plants as metadata.
    
    Returns
    -------
    Path
        Path to saved file
    
    Examples
    --------
    >>> save_snapshot(events, 'events', 'exp', 'sine_fit', plants=plants)
    >>> save_snapshot(plants, 'plants', 'sim', 'build')
    """
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fname = _build_snapshot_filename(file_type, data_type, stage, timestamp)
    path = CACHE_PATH / fname
    
    # If saving events with plants context, wrap them
    if file_type == "events" and plants is not None:
        obj = {"events": obj, "plants": plants}
    
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    
    print(f"Saved {file_type} {data_type} snapshot ({stage}) → {path}")
    return path


def load_snapshot(file_type, data_type, stage=None, path=None, 
                  extract_plants=False, plants=None, bind_context=False):
    """
    Load a snapshot with optional context binding.
    
    Parameters
    ----------
    file_type : str
        'events' or 'plants'
    data_type : str
        'exp', 'sim', 'xtrm', 'motor'
    stage : str, optional
        If None, loads latest matching file.
    path : str or Path, optional
        Explicit file path. If provided, ignores file_type/data_type/stage.
    extract_plants : bool
        If True and file_type='events' with plants metadata, return (events, plants).
        Otherwise return only the object.
    plants : list, optional
        Plant objects to bind context to events. Only used if file_type='events'.
    bind_context : bool
        If True and plants provided/available, bind plant context to events.
    
    Returns
    -------
    obj or tuple
        Loaded object, or (events, plants) if extract_plants=True
    
    Notes
    -----
    Events without plant context (e.g., simulation events) load normally.
    Context binding only occurs if plants are provided/available AND bind_context=True.
    
    Examples
    --------
    >>> events = load_snapshot('events', 'exp', stage='sine_fit')
    >>> events, plants = load_snapshot('events', 'exp', 'sine_fit', extract_plants=True)
    >>> events = load_snapshot('events', 'exp', 'sine_fit', plants=plants, bind_context=True)
    >>> sim_events = load_snapshot('events', 'sim', 'build')  # No plants needed
    >>> plants = load_snapshot('plants', 'xtrm', 'build')
    """
    if path is None:
        path = _get_latest_snapshot(file_type, data_type, stage)
    else:
        path = Path(path)
    
    with open(path, "rb") as f:
        obj = pickle.load(f)
    
    # Handle wrapped format (events + plants) — only for event snapshots
    if file_type == "events":
         # Pick the right Event class based on data_type
        if data_type=="exp": from src.core.exp_event import Event
        elif data_type=="motor": from src.core.motor_event import Motor_Event as Event
        elif data_type=="xtrm": from src.core.xtrm_event import Xtrm_Event as Event
        elif data_type=="sim": from src.core.sim_event import SimulationEvent as Event
        else: raise ValueError(f"Unknown data_type: {data_type}")

        if isinstance(obj, dict) and "events" in obj:
            # Wrapped format with metadata
            events = obj["events"]
            plants_from_snapshot = obj.get("plants")
        else:
            # Unwrapped format — plain events object (e.g., simulation)
            events = obj
            plants_from_snapshot = None
        
        # Resolve which plants to use for context binding
        available_plants = plants or plants_from_snapshot
        
        # Bind context if requested and plants available
        if bind_context and available_plants is not None:
            if isinstance(available_plants, dict):
                plant_registry = available_plants
            else:
                plant_registry = {p.exp_num: p for p in available_plants}

            ctx = PlantContext(plant_registry)
            Event.bind_context(ctx)
            print(f"Bound {len(events)} events to {len(plant_registry)} plants")
        elif bind_context:
            print(f"Note: No plants available for context binding (expected for simulation events)")
        
        if extract_plants and available_plants is not None:
            return events, available_plants
        return events
    
    return obj

