<script setup lang="ts">
import Projects from './pages/projects/Projects.vue'
</script>

<template>
  <main>
    <p class="eyebrow">MDAC2 / 本地制谱工作台</p>
    <h1>从音乐开始，<br />把每一拍交还给你。</h1>
    <Projects />
  </main>
</template>

<style>
:root { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #e8eaf0; background: #12151e; font-synthesis: none; }
body { margin: 0; }
main { max-width: 760px; margin: 0 auto; padding: clamp(40px, 12vw, 140px) 24px; }
.eyebrow { color: #a5b4fc; letter-spacing: .12em; font-size: .85rem; }
h1 { font-size: clamp(2rem, 5vw, 3.5rem); line-height: 1.4; font-weight: 600; }
.status { display: inline-block; padding: 8px 14px; border: 1px solid #4b5577; border-radius: 6px; }
.description { color: #b9c0d2; line-height: 1.8; max-width: 560px; }
</style>
