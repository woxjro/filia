import { assert } from 'console';
import * as esprima from 'esprima';
import * as estree from 'estree';
import * as fs from 'fs';
import * as js from './javascript';
import * as mlir from './mlir';
import * as mlirb from './mlir/basic';
import * as mlirjs from './mlir/javascript';
import * as mlirmichelson from './mlir/michelson';
import * as cf from './mlir/cf';
import { ReturnOp } from './mlir/standard';
import { BlockId, Block, BlockArgDecl, Op, Value } from './mlir';

type TypeEnv = Map<Value, mlir.TypeAttr>;

function compileStatement(
  stmt: estree.Directive | estree.Statement | estree.ModuleDeclaration,
) {
  switch (stmt.type) {
    // Statement or directive
    case 'ExpressionStatement':
      throw new Error('Not implemented:ExpressionStatement');
    case 'BlockStatement':
      throw new Error('Not implemented:BlockStatement');
    case 'StaticBlock':
      throw new Error('Not implemented:StaticBlock');
    case 'EmptyStatement':
      throw new Error('Not implemented:EmptyStatement');
    case 'DebuggerStatement':
      throw new Error('Not implemented:DebuggerStatement');
    case 'WithStatement':
      throw new Error('Not implemented:WithStatement');
    case 'ReturnStatement': {
      if (stmt.argument) {
        switch (stmt.argument.type) {
          case 'Identifier': {
            console.log(`return ${stmt.argument.name}`);
            break;
          }
          default:
            throw new Error(`Not implemented:${stmt.argument.type}`);
        }
      } else {
        throw new Error('Not implemented');
      }
      break;
    }
    case 'LabeledStatement':
      throw new Error('Not implemented:LabeledStatement');
    case 'BreakStatement':
      throw new Error('Not implemented:BreakStatement');
    case 'ContinueStatement':
      throw new Error('Not implemented:ContinueStatement');
    case 'IfStatement':
      throw new Error('Not implemented:IfStatement');
    case 'SwitchStatement':
      throw new Error('Not implemented:SwitchStatement');
    case 'ThrowStatement':
      throw new Error('Not implemented:ThrowStatement');
    case 'TryStatement':
      throw new Error('Not implemented:TryStatement');
    case 'WhileStatement':
      throw new Error('Not implemented:WhileStatement');
    case 'DoWhileStatement':
      throw new Error('Not implemented:DoWhileStatement');
    case 'ForStatement':
      throw new Error('Not implemented:ForStatement');
    case 'ForInStatement':
      throw new Error('Not implemented:ForInStatement');
    case 'ForOfStatement':
      throw new Error('Not implemented:ForOfStatement');
    // Declaration
    case 'FunctionDeclaration': {
      if (stmt.id?.name == 'smartContract') {
        console.log(stmt);
        compile(stmt.body.body);
      } else if (stmt.id?.name == 'MichelsonGetAmount') {
      } else if (stmt.id?.name == 'MichelsonMakePair') {
      } else if (stmt.id?.name == 'MichelsonMakeOperationList') {
      } else if (stmt.id?.name == 'MichelsonMakeResultPair') {
      }
      break;
    }
    case 'VariableDeclaration': {
      switch (stmt.kind) {
        case 'var':
          throw new Error('Not implemented');
        case 'let':
          throw new Error('Not implemented');
        case 'const':
          console.log(stmt.declarations[0]);
      }
      break;
    }
    case 'ClassDeclaration':
      throw new Error('Not implemented:ClassDeclaration');
  }
}

export function compile(
  stmts: (estree.Directive | estree.Statement | estree.ModuleDeclaration)[],
) {
  for (const stmt of stmts) {
    compileStatement(stmt);
  }
}
