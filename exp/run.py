import trlx
# from trlx.data.default_configs import default_ppo_config, TrainConfig

# 1. You can train a model using a reward function or a reward-labeled dataset.

# 1.1. Using a reward function
trainer = trlx.train('gpt2', reward_fn=lambda samples, **kwargs: [sample.count('cats') for sample in samples])

# # 1.2. Using a reward-labeled dataset
# trainer = trlx.train('EleutherAI/gpt-j-6B', samples=['dolphins', 'geese'], rewards=[1.0, 100.0])

# # 2. Using a prompt-completion dataset
# trainer = trlx.train('gpt2', samples=[['Question: 1 + 2 Answer:', '3'], ['Question: Solve this equation: ∀n>0, s=2, sum(n ** -s). Answer:', '(pi ** 2)/ 6']])

# # 3. Trainers provide a wrapper over their underlying model
# # trainer.generate(**tokenizer('Q: Who rules the world? A:', return_tensors='pt'), do_sample=True)

# # 4. Configure Hyperparameters
# from trlx.data.default_configs import default_ppo_config, TrainConfig

# config = default_ppo_config()
# config.model.model_path = 'EleutherAI/gpt-neox-20b'
# config.train.seq_length = 32
# config.train.batch_size = 16

# trainer = trlx.train(config=config, reward_fn=lambda samples, **kwargs: [float(int(sample)) for sample in samples])

# # 5. Save the resulting model to a Hugging Face pretrained language model. (Ready to upload to the Hub!)
# trainer.save_pretrained('/path/to/output/folder/')
