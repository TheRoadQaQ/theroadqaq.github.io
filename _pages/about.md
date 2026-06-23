---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<head>
    <style>
	:root {
	  --theme-color: #4A90D9;
	  --venue-bg-color: rgb(108, 149, 181);
	}
	    
	g {
		color: #aaaaaa
	}

	 pt {
		color: var(--title-color);
		font-weight: 500;
	}

	 em {
		font-style: italic;
	}

	 venue {
		background-color: #4A90D9;
		color: #ffffff;
		font-size: 70%;
		font-weight: bold;
		line-height: 170%;
		margin-right: 0.25em;
		width: 5em;
		display:inline-block;
		text-align: center;
		border-width: 0px;
		border-style: none;
		border-radius: 0.1rem;
		height: 1.7em;
		vertical-align:text-bottom;
		margin-bottom: 0.1em;
	}

	 venue1 {
		background-color: var(--venue-bg-color);
		color: #ffffff;
		font-size: 70%;
		font-weight: bold;
		line-height: 170%;
		margin-right: 0.25em;
		width: 5em;
		display:inline-block;
		text-align: center;
		border-width: 0px;
		border-style: none;
		border-radius: 0.1rem;
		height: 1.7em;
		vertical-align:text-bottom;
		margin-bottom: 0.1em;
	}
 
	.filter {
		color: var(--color);
		background-color: #fff;
		border: var(--border);
		border-style: solid;
		border-radius: 0.2rem;
		border-width: 1.5px;
		transition: all .3s;
		touch-action: manipulation;
		font-size: 80%;
		line-height: 120%;
	}
	
	.filter:focus {
		color: #171e29;
	}
	  
	  .filter:hover {
		border-color: var(--theme-color);
		color: white;
		background-color: var(--theme-color);
		fill: var(--theme-color);
	  }
	  
	  .filter:active {
		border-color: var(--theme-color);
		color: var(--theme-color);
		fill: var(--theme-color);
	  }
	  
	.button-59 {
	  align-items: center;
	  background-color: #fff;
	  border: 1px solid #dadada;
	  box-sizing: border-box;
	  color: #000000;
	  cursor: pointer;
	  display: inline-block;
	  fill: #000;
	  font-family: 'Nunito';
	  font-size: 0.7rem;
	  height: 1.1rem;
	  justify-content: center;
	  line-height: 1.3;
	  min-width: 60px;
	  outline: 0;
	  padding: 0 10px;
	  text-align: center;
	  text-decoration: none;
	  transition: color .3s, background-color .3s, border-color .3s;
	  user-select: none;
	  -webkit-user-select: none;
	  touch-action: manipulation;
	  margin-right: 0.2em;
	  border-radius: 0.2rem;
	}
	
	.button-59:hover {
	  border-color: var(--theme-color);
	  color: #fff;
	  fill: var(--theme-color);
	  background-color: var(--theme-color);
	  text-decoration: none;
	}
	
	.button-59:active {
	  border-color: var(--theme-color);
	  color: #fff;
	  fill: var(--theme-color);
	  background-color: var(--theme-color);
	}
	
	@media (min-width: 768px) {
	  .button-59 {
	    padding-left: 5px;
	    padding-right: 5px;
	  }
	}
    </style>
    <script>
        function checkFilter(type, li) {
            if (type == "All") {
                return true
            }
            else if (type == "First-authored") {
                res = li.getAttribute("first_authored")
                return res
            }
            else {
                cate = li.getAttribute("category")
                if (!cate) {
                    return false
                }
                items = cate.split(',')
                for (j = 0; j < items.length; j++) {
                    if (type.toUpperCase() == items[j].toUpperCase()) {
                        return true
                    }
                }
                return false
            }
        }

        function filterPub(type) {
            ul = document.getElementById("publications")
            li = ul.getElementsByTagName("li")
            for (i = 0; i < li.length; i++) {
                if (!checkFilter(type, li[i])) {
                    li[i].style.display = "none";
                }
                else {
                    li[i].style.display = ""
                }
            }
            bts = document.getElementsByClassName("filter")
            for (k = 0; k < bts.length; k++) {
                if (bts[k].textContent == type) {
                    bts[k].style.setProperty("--color", "#000")
                    bts[k].style.setProperty("--border", "#000")
                }
                else {
                    bts[k].style.setProperty("--color", "#a0a0a0")
                    bts[k].style.setProperty("--border", "#d3d3d3")
                }
            }
        }
    </script>
</head>

<span class='anchor' id='about-me'></span>

# About Me

I am a Master student at **Peking University**, advised by Professor [Bin Cui](https://cuibinpku.github.io/) and Assistant Professor [Wentao Zhang](https://zwt233.github.io/). My research interests include **large language models**, **AI agents**, and **data scaling**.

Before this, I obtained my bachelor's degree in Computer Science from **Renmin University of China**. Previously, I conducted my internship at **Tencent AI Lab** and **Meituan LongCat**. I am currently a research intern at **Alibaba Qwen**.

Please feel free to reach out if you are interested in collaboration.

<br>

# Educations
- **Peking University** (2024 - Present), Master Student in Computer Science
- **Renmin University of China** (2020 - 2024), B.Sc. in Computer Science

<br>

# Selected Projects

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2026</div><img src='images/relift.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions**

- **Lu Ma**, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, Bin Cui, Wentao Zhang
- ICLR 2026.
- ReLIFT combines reinforcement learning and supervised fine-tuning in an interleaved training framework that expands LLM reasoning capabilities beyond the limits of RL alone.
- [[Paper]](https://arxiv.org/pdf/2506.07527) [[Code]](https://github.com/TheRoadQaQ/ReLIFT)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ACL 2026</div><img src='images/leash.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model**

- Yanhao Li\*, **Lu Ma\***, Jiaran Zhang, Lexiang Tang, Wentao Zhang, Guibo Luo
- ACL 2026.
- LEASH introduces an adaptive length-penalty mechanism for LLM reasoning using a Lagrangian primal-dual optimization framework.
- [[Paper]](https://arxiv.org/pdf/2512.21540)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Open Source</div><img src='images/dataflow.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation**

- Hao Liang, ..., **Lu Ma**, ..., Bin Cui, Wentao Zhang
- DataFlow is a data preparation system designed to parse, generate, process, and evaluate high-quality data from noisy sources, improving the performance of LLMs in specific domains.
- [[Paper]](https://arxiv.org/abs/2512.16676) [[Code]](https://github.com/OpenDCAI/DataFlow)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Open Source</div><img src='images/dataflex.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**DataFlex: A Dynamic Training Framework**

- DataFlex is an advanced dynamic training framework built on top of LLaMA-Factory. 
- [[Code]](https://github.com/OpenDCAI/DataFlex)
</div>
</div>

<br>

# Publications

<div class="pub-controls">
  <p>*: Equal Contribution &nbsp; †: Equal Contribution<br><br></p>

  <button class="filter" type="button" onclick="filterPub('All')" style="--color: #000; --border: #000">All</button>
  <button class="filter" type="button" onclick="filterPub('First-authored')">First author</button>
</div>

<ul id="publications">
  <li first_authored=true category="LLM">
    <venue>ICLR'26</venue><pt>Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions</pt><br>
    <b>Lu Ma</b><g>, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Yanhao Li, Bin Cui, Wentao Zhang</g> <br />
    International Conference on Learning Representations<br />
    <p>
      <a href="https://arxiv.org/pdf/2506.07527" class="button-59">PDF</a>
      <a href="https://github.com/TheRoadQaQ/ReLIFT" class="button-59">Code</a>
    </p>
  </li>
  <li first_authored=true category="LLM">
    <venue>ACL'26</venue><pt>Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model</pt><br>
    <g>Yanhao Li†, </g><b>Lu Ma†</b><g>, Jiaran Zhang, Lexiang Tang, Wentao Zhang, Guibo Luo</g> <br />
    Association for Computational Linguistics<br />
    <p>
      <a href="https://arxiv.org/pdf/2512.21540" class="button-59">PDF</a>
    </p>
  </li>
  <li category="VLM">
    <venue>CVPR'25</venue><pt>Native Visual Understanding: Resolving Resolution Dilemmas in Vision-Language Models</pt><br>
    <g>Junbo Niu, Yuanhong Zheng, Ziyang Miao, Hejun Dong, Chunjiang Ge, Hao Liang, </g><b>Lu Ma</b><g>, Bohan Zeng, Qiahao Zheng, Conghui He, Wentao Zhang</g> <br />
    IEEE/CVF Conference on Computer Vision and Pattern Recognition<br />
    <p>
      <a href="https://arxiv.org/abs/2506.12776" class="button-59">PDF</a>
    </p>
  </li>
  <li category="LLM">
    <venue>AAAI'25</venue><pt>Not All Tokens and Heads Are Equally Important: Dual-Level Attention Intervention for Hallucination Mitigation</pt><br>
    <g>Lexiang Tang, Xianwei Zhuang, Bang Yang, Zhiyuan Hu, Hongxiang Li, </g><b>Lu Ma</b><g>, Jinghan Ru, Yuexian Zou</g> <br />
    AAAI Conference on Artificial Intelligence<br />
    <p>
    </p>
  </li>
  <li first_authored=true category="GNN">
    <venue>TKDE'25</venue><pt>Acceleration Algorithms in GNNs: A Survey</pt><br>
    <b>Lu Ma</b><g>, Zeang Sheng, Xunkai Li, Xinyi Gao, Zhezheng Hao, Ling Yang, Xiaonan Nie, Jiawei Jiang, Wentao Zhang, Bin Cui</g> <br />
    IEEE Transactions on Knowledge and Data Engineering<br />
    <p>
    </p>
  </li>
  <li first_authored=true category="LLM">
    <venue1>arXiv'26</venue1><pt>Training with Harnesses: On-Policy Harness Self-Distillation for Complex Reasoning</pt><br>
    <g>Zhengyang Zhao†, </g><b>Lu Ma†</b><g>, Wentao Zhang</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2605.08741" class="button-59">PDF</a>
    </p>
  </li>
  <li category="LLM">
    <venue1>arXiv'26</venue1><pt>ANDES: Agent Native Data Evolving Synthesis Tool for Autonomous Instruction Alignment</pt><br>
    <g>Zhengyang Zhao, Shengjie Ye, </g><b>Lu Ma</b><g>, Hao Liang, Hengyi Feng, Wentao Zhang</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2606.01279" class="button-59">PDF</a>
    </p>
  </li>
  <li first_authored=true category="LLM">
    <venue1>arXiv'26</venue1><pt>Thinking by Subtraction: Confidence-Driven Contrastive Decoding for LLM Reasoning</pt><br>
    <g>Lexiang Tang, Weihao Gao, Bingchen Zhao, </g><b>Lu Ma</b><g>, Bang Yang, Yuexian Zou</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2602.18232" class="button-59">PDF</a>
    </p>
  </li>
  <li first_authored=true category="LLM">
    <venue1>arXiv'26</venue1><pt>GIFT: Unlocking Global Optimality in Post-Training via Finite-Temperature Gibbs Initialization</pt><br>
    <g>Zhengyang Zhao†, </g><b>Lu Ma†</b><g>, Yizhen Jiang, Xiaochen Ma, Zimo Meng, Chengyu Shen, Lexiang Tang, Haoze Sun, Peng Pei, Wentao Zhang</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2601.09233" class="button-59">PDF</a>
    </p>
  </li>
  <li category="LLM">
    <venue1>arXiv'25</venue1><pt>DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation</pt><br>
    <g>Hao Liang, Xiaochen Ma, ..., </g><b>Lu Ma</b><g>, ..., Bin Cui, Wentao Zhang</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2512.16676" class="button-59">PDF</a>
      <a href="https://github.com/OpenDCAI/DataFlow" class="button-59">Code</a>
    </p>
  </li>
  <li first_authored=true category="LLM">
    <venue1>arXiv'25</venue1><pt>DARO: Difficulty-Aware Reweighting Policy Optimization</pt><br>
    <g>Jingyu Zhou†, </g><b>Lu Ma†</b><g>, Hao Liang, Chengyu Shen, Bin Cui, Wentao Zhang</g> <br />
    <p>
      <a href="https://arxiv.org/abs/2510.09001" class="button-59">PDF</a>
    </p>
  </li>
</ul>

<br>

<span class='anchor' id='internships'></span>

# Internships
- **Tencent AI Lab**, Research Intern
- **Meituan LongCat**, Research Intern
- **Alibaba Qwen**, Research Intern
