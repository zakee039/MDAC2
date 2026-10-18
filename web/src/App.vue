<script setup lang="ts">
import {ref} from 'vue'
import AutoStudio from './pages/projects/AutoStudio.vue'
import brandIcon from './assets/miku-cute.png'
import StudioIcon from './components/StudioIcon.vue'
import {t} from './studioI18n'
const manager=ref(false),settings=ref(false)
</script>
<template><div class="minimal-studio"><header class="studio-header"><div class="studio-header-inner"><button class="header-action" :aria-label="t('projects')" @click="manager=true"><StudioIcon name="folder" :size="23"/><span>{{t('projects')}}</span></button><a class="studio-brand" href="#workflow" aria-label="MDAC · Muse Dash AI Chart"><img :src="brandIcon" alt=""/><div><strong>MDAC</strong><small>Muse Dash AI Chart</small></div></a><button class="header-action settings-action" :aria-label="t('settings')" @click="settings=true"><StudioIcon name="settings" :size="23"/><span>{{t('settings')}}</span></button></div></header><main class="studio-main"><AutoStudio :manager-open="manager" :settings-open="settings" @close-manager="manager=false" @close-settings="settings=false"/></main></div></template>
<style scoped>
.studio-header{border-bottom:1px solid #e1e8df;background:#ffffffd9}.studio-header-inner{height:82px;max-width:1180px;padding:0 30px;margin:auto;display:grid;grid-template-columns:1fr auto 1fr;align-items:center}.header-action{justify-self:start;border:0;background:transparent;gap:10px;padding:12px;color:#4a6558;font-size:13px}.settings-action{justify-self:end}.studio-brand{display:flex;align-items:center;gap:12px;color:#2e4b3c}.studio-brand strong{font-size:28px;letter-spacing:1px}.studio-brand img{width:48px;height:48px;object-fit:contain}.studio-brand>div{display:flex;flex-direction:column;align-items:center;gap:3px}.studio-brand small{font-size:10px;letter-spacing:1.3px;color:#6b8173}.studio-main{max-width:1180px;padding:28px 30px 60px;margin:auto}@media(max-width:600px){.studio-header-inner{height:76px;padding:0 16px}.studio-brand img{width:40px;height:40px}.studio-brand small{font-size:9px}.header-action span{display:none}.studio-brand strong{font-size:24px}.studio-main{padding:20px 16px 40px}}
</style>
