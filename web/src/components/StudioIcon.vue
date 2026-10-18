<script setup lang="ts">
defineProps<{ name: string; size?: number }>()
const paths: Record<string, string> = {
 settings: 'M9 3h6l1 3 3 1 2 5-2 5-3 1-1 3H9l-1-3-3-1-2-5 2-5 3-1 1-3ZM16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
 wave: 'M3 10v4m4-8v12m5-16v20m5-16v12m4-8v4',
 grid: 'M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z',
 folder: 'M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z',
 sliders: 'M4 3v7m0 4v7M12 3v11m0 4v3M20 3v3m0 4v11M1 10h6m2 8h6m2-12h6',
 arrow: 'M5 12h14m-5-5 5 5-5 5',
 plus: 'M12 5v14M5 12h14',
 upload: 'M12 16V3m-5 5 5-5 5 5M3 16v4a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1v-4',
 check: 'm5 12 4 4L19 6',
 lock: 'M6 10h12v11H6zM8 10V6a4 4 0 0 1 8 0v4M12 14v3',
 book: 'M12 5v16M12 5C9 2 5 2 2 3v16c3-1 7-1 10 2 3-3 7-3 10-2V3c-3-1-7-1-10 2Z',
 headphones: 'M3 14v-3a9 9 0 0 1 18 0v3M3 12h3v9H3a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1Zm18 0h-3v9h3a1 1 0 0 0 1-1v-7a1 1 0 0 0-1-1Z',
 clock: 'M12 7v6l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
 layers: 'm12 3 10 5-10 5L2 8l10-5ZM2 12l10 5 10-5M2 16l10 5 10-5',
}
</script>
<template><svg :width="size || 20" :height="size || 20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path :d="paths[name] || paths.wave" /></svg></template>
