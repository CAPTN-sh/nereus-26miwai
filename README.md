# Context-aware probabilistic ship trajectory forecasting

## Requirements
To install all the requirements, one needs to first install:
+ conda
+ poetry
+ flower

A detailed list of the required libraries can be found in:
``poetry.toml``

The proper installation must then be done with poetry and conda.

## Structure
The project is structured in a way that allows for easy addition of new pipelines and tasks. 
### ``preprocessing`` branch 
The main components are:
```bash
.
|-- data
|   `-- kiel
|       |-- ais
|       |   |-- 1_raw
|       |   |-- 2_decoded
|       |   `-- 3_features
|       |-- maps
|       |   |-- kiel_buoys.geojson
|       |   |-- kiel_fjord_epsg4326.geojson
|       |   `-- kiel_marinas.geojson
|       `-- ship_db
|           `-- ship_type_dict.yaml
|-- .flake8
|-- .gitignore
|-- .history
|-- images
|-- Makefile
|-- playground
|-- pyproject.toml
|-- README.md
|-- src
|   |-- main.py
|   |-- preprocessing
|   |   |-- configs
|   |   |   |-- decode.yaml
|   |   |   |-- _main.yaml
|   |   |   |-- transform_edges.yaml
|   |   |   `-- transform_nodes.yaml
|   |   |-- steps
|   |   |   |-- decoding
|   |   |   |   |-- decoder.py
|   |   |   |   `-- pipeline.py
|   |   |   `-- transform
|   |   |       |-- edges
|   |   |       |   |-- metrics.py
|   |   |       |   |-- pipeline.py
|   |   |       |   `-- ship.py
|   |   |       `-- nodes
|   |   |           |-- geo_features.py
|   |   |           |-- interpolate.py
|   |   |           |-- pipeline.py
|   |   |           |-- trajectory.py
|   |   |           `-- wrapper.py
|   |   `-- utils
|   |       |-- df_transformer.py
|   |       |-- pipeline
|   |       |   |-- function_inport.py
|   |       |   |-- pipeline_executor.py
|   |       |   `-- pipeline.py
|   |       `-- ship_info_system
|   |           |-- ship_info.py
|   |           `-- webcrawler.py
|   `-- utils
|       |-- config.py
|       `-- map_reader.py
|-- tools.yml
`-- _version.py
```

### Description of Core Modules

#### 🧠 `main.py` + `_main.yaml`
- Entry point to the preprocessing system.
- Loads pipeline definitions from `_main.yaml` and initializes processing steps accordingly.

#### ⚙️ `pipeline_executor.py`
- Coordinates **multiprocessing** for task execution.
- Iteratively loads tasks from YAML configuration and executes them in **parallel**.

---

### Pipelines

#### 🔍 `decoding/pipeline.py` + `decode.yaml` → `decoder.py`
- Decodes raw AIS data and stores results in daily tables.
- Configured via `decode.yaml`.
- Output is structured, cleaned, and stored per day.

#### 📌 `transform/nodes/pipeline.py` + `transform_nodes.yaml` → `trajectory.py`
- Extracts classic features from individual ship trajectories.
- `trajectory.py` is the core logic module:
  - Supports arbitrary functions (must take a trajectory and return a trajectory).
  - Supports built-in methods from `movingpandas.Trajectory`.
- Enriches data with web-scraped ship info via `ship_info_system`.

#### 🧭 `transform/edges/pipeline.py` + `transform_edges.yaml` → `ship.py`
- Computes **pairwise features** between ships within a given proximity (e.g. 1000m).
- `ship.py` contains feature logic.
- Currently hardcoded; planned to be generalized via config like nodes.

---

### Utilities

#### 🧰 `utils/pipeline/`
- Tools for dynamic function import (`function_inport.py`) and generic pipeline infrastructure (`pipeline.py`).
- `pipeline_executor.py` uses these components to orchestrate execution.

#### 🌐 `utils/ship_info_system/`
- Fetches vessel metadata from external web sources.
- Contains scraping and ship info logic.

#### 🧾 `df_transformer.py`
- General-purpose transformer functions for working with pandas dataframes in the pipelines.

#### 📍 `map_reader.py`
- Loads and manages static geo-referenced data (e.g., buoys, marinas, regions) used in preprocessing.

<!-- # structure / run

# main.py + _main.yaml
    main.py starts up the pipelines and loads the config-files written in _main.yaml

# - - pipeline_executor.py
    the pipeline_executor.py is responsable for multiprocessing
    the tasks from the pipelines loaded only iterative and processed in parallel

# - - - - decoding.pipeline.py + decode.yaml -> decoder.py
    the decoding pipeline decodes the ais data and saves it in tables according to decode.yaml
    -> one file per day & table

# - - - - transform.nodes.pipeline.py + transform_nodes.yaml -> trajectory.py
    the nodes pipeline extracts all the "classic" features of the trajectories
    trajectory.py has the main logic
    all steps can be defined in the config:
        - function: any function 
            - has to take in trajectory + args 
            - has to return trajectory
        - method: any method from movingpandas.Trajectory
    ship info from the web is added right at the end (utils.ship_info_system)

# - - - - transform.edges.pipeline.py + transform_edges.yaml -> ship.py
    the edges pipeline calculates the features between all ships in proximity (1000m)
    ship.py has the main logic
    the config is not fleshed out yet. just hardcode new stuff for now -->