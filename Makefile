test: hawkZip
		./hawkZip -i ./1800x3600/TAUX_1_1800_3600.f32 -e 1e-4
		
hawkZip: hawkZip_main.c
		gcc hawkZip_main.c -O0 -o hawkZip -lm -fopenmp