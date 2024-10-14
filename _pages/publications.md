---
layout: page
permalink: /publications/
title: publications
description: 
nav: true
nav_order: 2
---
<!-- _pages/publications.md -->
<h3>Papers</h3>
<div class="publications">
    {% bibliography -f {{ site.scholar.bibliography }} --query !@inproceedings %}

</div>

<h3>Workshops</h3>
<div class="publications">
    {% bibliography -f {{ site.scholar.bibliography }} --query @inproceedings[]* %}

</div>
