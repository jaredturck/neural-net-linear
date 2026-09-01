import neuralnet as nn


if __name__ == '__main__':
    tokenizer = nn.Tokenizer.train('train.txt', vocab_size=320)

    model = nn.GPT(
        tokenizer.vocab_size,
        context_length=32,
        embedding_dim=64,
        heads=4,
        layers=2,
        hidden_dim=192,
        seed=42,
    )

    optimizer = nn.AdamW(
        learning_rate=0.002,
        weight_decay=0.01,
    )

    print('[+] Training GPT', flush=True)
    model.train(
        tokenizer,
        'train.txt',
        epochs=20,
        batch_size=8,
        steps_per_epoch=10,
        optimizer=optimizer,
        warmup_steps=10,
        grad_clip=1.0,
        seed=42,
        log_every=20,
    )

    print(model.generate(
        tokenizer,
        'spider,',
        max_tokens=64,
        temperature=0.8,
        top_k=20,
        seed=42,
    ))

    model.free()
    tokenizer.free()
