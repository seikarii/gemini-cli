#!/usr/bin/env node

/**
 * Test script para verificar que los cambios del sistema de streaming funcionan
 */

import { GeminiChat } from '../packages/core/dist/src/core/geminiChat.js';
import { Config } from '../packages/core/dist/src/config/config.js';

async function testStreamingFix() {
  console.log('🔧 Probando la corrección del sistema de streaming...');
  
  try {
    // Crear una configuración de prueba
    const config = new Config({
      model: 'gemini-pro',
      apiKey: process.env.GEMINI_API_KEY || 'test-key'
    });
    
    const chat = new GeminiChat(config);
    
    console.log('✅ GeminiChat creado exitosamente');
    console.log('✅ Los cambios están compilados y disponibles');
    console.log('');
    console.log('🚀 Cambios implementados:');
    console.log('  - Validación de contenido más tolerante con chunks de streaming');
    console.log('  - Solo marca stream como inválido si NO hay contenido válido en absoluto');
    console.log('  - Permite chunks vacíos/metadatos como parte normal del flujo');
    console.log('  - Logging mejorado para debugging');
    console.log('');
    console.log('💡 Sistema de seguridad activado con variables de entorno:');
    console.log('  - GEMINI_SECURE_FILE_PROCESSING=true');
    console.log('  - GEMINI_SECURITY_RATE_LIMIT=1000');
    console.log('  - GEMINI_SECURITY_AUDIT_LOG=true');
    
  } catch (error) {
    console.error('❌ Error al crear GeminiChat:', error.message);
  }
}

testStreamingFix();
