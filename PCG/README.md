# Procedural Content Generation for Maps

Generate a random bases+workers style map (terrain + two players + workers). By default the XML is written where **gym** and the Java GUI expect it:

`gym_microrts/microrts/maps/pcg/pcg_generated.xml`

```bash
python PCG/pcg.py --width 16 --height 16 --seed 42
```

Optional: `--out /path/to/file.xml` to choose another path (still under `gym_microrts/microrts/maps/` if you want gym/GUI to load it by relative name).

**Single-gene GA / simulations:** set the map env to that file, e.g.

`SINGLE_GENE_MAP_PATH=maps/pcg/pcg_generated.xml`

(Regenerate with `--seed` for reproducibility, or without `--seed` for a new layout each run.)

You may use microrts's GUI editor at `gym_microrts/microrts/src/gui/frontend/FrontEnd.java` to visualize the map.

```
bash build.sh
java -cp gym_microrts/microrts/microrts.jar gui.frontend.FrontEnd
```
