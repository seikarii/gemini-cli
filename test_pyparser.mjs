/* eslint-env node */
import pkg from 'pyparser';
const { parse } = pkg;

async function testPythonAST() {
  try {
    console.log('Testing pyparser...');
    const ast = await parse('def test_function():\n    return "hello"');
    console.log('AST parsing successful:', !!ast);
    console.log('AST keys:', Object.keys(ast));
    console.log('AST structure:', JSON.stringify(ast, null, 2));
  } catch (error) {
    console.error('Error:', error);
  }
}

testPythonAST();
