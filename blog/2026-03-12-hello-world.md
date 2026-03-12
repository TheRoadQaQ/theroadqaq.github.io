---
layout: post
title: Hello World
date: 2026-03-12 12:00:00
description: My first blog post — a starting point for sharing thoughts on research, AI, and beyond.
tags: life research
categories: thoughts
img: assets/img/roadma.jpg
---

This is my first blog post. I'm a Master's student at Peking University working on large language models, AI agents, and reinforcement learning.

This blog is a place where I'll share:

- **Research notes** — ideas, paper summaries, and things I'm exploring
- **Learning logs** — lessons from experiments, mistakes, and progress
- **Random thoughts** — anything else worth writing down

---

### How to add a new post

Just create a new `.md` file inside the `blog/` folder. The front matter at the top controls how the post appears in the card list:

```yaml
---
layout: post
title: Your Post Title
date: 2026-03-12 12:00:00
description: A short description shown on the card.
tags: tag1 tag2
categories: your-category
img: assets/img/your-thumbnail.jpg   # optional – shows on the card
---
```

That's it — no config changes needed.

---

### Math support

Inline math: \(E = mc^2\)

Block math:

$$
\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} r_t \right]
$$

### Code support

```python
def hello_world():
    print("Hello, World!")
```

---

More posts coming soon.

