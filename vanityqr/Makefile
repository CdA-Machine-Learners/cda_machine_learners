run: build
	python main.py


build:
	g++ -shared -o core/worker.so core/post_proc_worker.cpp -lpython3 -lm -I venv/include/python3.11 -I venv/lib/python3.11/site-packages/numpy/core/include -L venv/lib/python3.11 -fPIC
