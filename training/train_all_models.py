from training.train_main_model import train_task


def main():
	for task_type in ["binary", "multiclass"]:
		train_task(task_type)


if __name__ == "__main__":
	main()
