---
layout: page
permalink: /publications/
title: publications
description: 
years: [2026, 2025, 2024, 2022, 2021, 2020, 2019, 2017, 2016, 2015, 2014, 2012, 2006]
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<p>† means equal contribution</p>

<div class="publications">

{%- for y in page.years %}
  {%- capture bib_count %}{% bibliography_count -f papers -q @*[year={{y}}]* %}{% endcapture %}
  {%- if bib_count != "0" %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
  {%- endif %}
{% endfor %}

</div>
