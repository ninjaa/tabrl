import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import fs from 'fs'
import path from 'path'

// Custom plugin to serve mujoco_wasm files
const mujocoWasmPlugin = () => {
  return {
    name: 'mujoco-wasm-plugin',
    configureServer(server) {
      server.middlewares.use('/mujoco_wasm', (req, res, next) => {
        try {
          const filePath = path.join(__dirname, 'mujoco_wasm', req.url)
          
          // Check if file exists
          if (!fs.existsSync(filePath)) {
            return next()
          }
          
          // Set appropriate headers based on file type
          if (req.url.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm')
          } else if (req.url.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript')
          } else if (req.url.endsWith('.d.ts')) {
            res.setHeader('Content-Type', 'text/plain')
          }
          
          // Add CORS headers
          res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless')
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
          
          // Serve the file
          const fileContent = fs.readFileSync(filePath)
          res.end(fileContent)
        } catch (error) {
          console.error('Error serving mujoco_wasm file:', error)
          next()
        }
      })
    }
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), mujocoWasmPlugin()],
  server: {
    port: 5173,
    host: true,
    headers: {
      'Cross-Origin-Embedder-Policy': 'credentialless',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
    fs: {
      // Allow serving files outside of root
      allow: ['..', './mujoco_wasm']
    }
  },
  optimizeDeps: {
    exclude: ['@webcontainer/api']
  },
  // Include WASM files as assets  
  assetsInclude: ['**/*.wasm', '**/*.data'],
  // Define public directory
  publicDir: 'public'
})
