---
layout: page
permalink: /teaching/
title: teaching
description: Lectures and (maybe in the future) courses that I have (co-)taught 
nav: true
nav_order: 6
horizontal: true
---

<!-- pages/projects.md -->
<div class="teaching">
<!-- Display projects without categories -->
  {%- assign sorted_teaching = site.teaching | sort: "importance" -%}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-1">
    {%- for teaching in sorted_teaching -%}
      {% include teaching.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for teaching in sorted_teaching -%}
      {% include teaching.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
</div>
