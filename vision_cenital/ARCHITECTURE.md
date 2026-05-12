# Arquitectura del Sistema de Visión Cenital — CargaBot

Este submódulo provee capacidades de localización global, mapeo de obstáculos y planificación de trayectorias en tiempo real utilizando una cámara HD cenital fija.

## Flujo de Nodos y Tópicos ROS 2

El sistema expone el nodo principal `overhead_coordinator_node` que orquesta la ejecución del pipeline:

- **Subscripciones:**
  - `/cargabot/goal_pose` (`geometry_msgs/msg/PoseStamped`): Destino ordenado por el operador o planificadores de alto nivel.
- **Publicaciones:**
  - `/cargabot/global_path` (`nav_msgs/msg/Path`): Camino métrico absoluto visualizable nativamente en RViz2.
  - `/cargabot/cmd_goto` (`geometry_msgs/msg/PoseStamped`): Coordenada incremental de avance (Lookahead) calculada con heurística de tolerancia a desviación.
  - `/cargabot/overhead_debug_video` (`sensor_msgs/msg/Image`): Previsualización de depuración con superposiciones de datos espaciales.

## Ciclo de Vida del Frame
1. **Captura:** Inyección VA-API de video comprimido H.264 por hardware a 1280x720 nativos.
2. **Corrección Proyectiva:** Eliminación de distorsión radial y proyección métrica de la pista completa.
3. **Mapeo:** Extracción de capas semánticas (Obstáculos + Cubos RGB) e inyección en matriz dispersa de costos de 5 cm x 5 cm.
4. **Búsqueda:** Bucle A* sobre la grilla de espacio libre.