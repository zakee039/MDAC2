<script setup lang="ts">
import {ref,watch,nextTick,onBeforeUnmount} from 'vue'
const props=defineProps<{open:boolean;title:string}>()
const emit=defineEmits<{close:[]}>()
const dialog=ref<HTMLDialogElement|null>(null)
watch(()=>props.open,async open=>{await nextTick();if(open&&!dialog.value?.open)dialog.value?.showModal();else if(!open&&dialog.value?.open)dialog.value.close()},{immediate:true,flush:'post'})
onBeforeUnmount(()=>dialog.value?.close())
</script>
<template><dialog ref="dialog" :aria-label="title" @cancel.prevent="emit('close')" @click="($event.target===dialog)&&emit('close')"><div class="dialog-body"><slot/></div></dialog></template>
<style scoped>
dialog{padding:0;border:1px solid #dce5db;border-radius:16px;width:min(680px,calc(100vw - 32px));max-height:85vh;background:#fff;color:#263e32;box-shadow:0 20px 80px #19302126}dialog::backdrop{background:#203b3659;backdrop-filter:blur(3px)}.dialog-body{padding:28px}@media(max-width:600px){.dialog-body{padding:20px}}
</style>
