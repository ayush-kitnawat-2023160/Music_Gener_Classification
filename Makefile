# Variables
SPECTROGRAM_DIR=Spectrogram
PLOT_DIR=Plot
MODEL_FILE=best_model.pth
LOG_FILE=log.txt
PYTHON_FILE=main.py

clean:
	rm -rf $(SPECTROGRAM_DIR) $(PLOT_DIR)
	rm -f $(MODEL_FILE) $(LOG_FILE)
	@echo "Cleaned up generated files."

run:
	mkdir -p $(SPECTROGRAM_DIR) $(PLOT_DIR)
	python $(PYTHON_FILE)

.PHONY: clean run