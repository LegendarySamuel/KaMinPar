mkdir -p logs/dKaMinPar-AGLP
cd logs/dKaMinPar-AGLP && rm -rf youtube_P1x32x1_seed1_eps0.03_k2.log && touch youtube_P1x32x1_seed1_eps0.03_k2.log
cd ../..
nohup mpirun -n 32 --bind-to core --map-by socket:PE=1 --report-bindings  ../KaMinPar/build/apps/dKaMinPar -k 2 --c-global-clustering-algorithm ag-lp -e 0.03 --seed=1 -G /global_data/graphs/benchmark_sets/seemaier/tuning/youtube.graph -t 1 -T >> /home/gil/projects/KaMinPar/logs/dKaMinPar-AGLP/youtube_P1x32x1_seed1_eps0.03_k2.log 2>&1 
