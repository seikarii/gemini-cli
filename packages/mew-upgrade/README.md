# mew-upgrade — notas de bundling

Breve nota sobre la convención de imports en este paquete:

- El código fuente de la UI vive en archivos .tsx (por ejemplo `MewApp.tsx`).
- Para el bundling, el script de build usa esbuild con la opción `--loader:.js=jsx`.
  Esto permite que la entrada (`src/app/web.tsx`) importe `./MewApp.js` — esbuild
  tratará ese import como JSX y resolverá la implementación escrita en `.tsx`.

Razón técnica:
- El repositorio usa resolución de módulos ESM tipo `node16`/`nodenext`. Con esa
  configuración TypeScript y Node exigen extensiones explícitas en los imports ESM
  (por ejemplo `./MewApp.js`). El pipeline aquí aprovecha esa convención y le indica
  a esbuild que trate `.js` como JSX para que el bundling funcione sin pasos
  adicionales de precompilación.

Recomendaciones para mantenedores:
- No elimines la extensión `.js` en imports dentro de `packages/mew-upgrade/src/app`.
- Si prefieres evitar esta convención, considera una de estas alternativas:
  - Cambiar el pipeline para compilar primero TS/TSX a JS y luego apuntar los imports
    a los artefactos compilados.
  - Usar un shim (archivo `.js` que re-exporte) — menos recomendado porque
    complica el flujo y las herramientas.

Esta nota está aquí para evitar regresiones accidentales: la importación `./MewApp.js`
es intencional y necesaria con la configuración de bundling actual.
