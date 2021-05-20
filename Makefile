
prepare:
	cd tasks && python prepare.py

clean:
	rm -rf ./data/processed/verify/*
	rm -rf ./data/processed/image/*
	rm -rf ./data/processed/labels/*
