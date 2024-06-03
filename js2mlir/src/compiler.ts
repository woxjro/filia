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

export interface CompileStatementResult {
  op: Op | null;
  typeEnv: TypeEnv;
}

function compileStatement(
  stmt: estree.Directive | estree.Statement | estree.ModuleDeclaration,
  typeEnv: TypeEnv,
): CompileStatementResult {
  switch (stmt.type) {
    // Statement or directive
    /*
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
    */
    case 'ReturnStatement': {
      if (stmt.argument) {
        switch (stmt.argument.type) {
          case 'Identifier': {
            const type = typeEnv.get(stmt.argument.name) as mlir.TypeAttr;
            const op = new ReturnOp([{ value: stmt.argument.name, type }]);
            console.log(op.toString());
            return { op, typeEnv };
          }
          default:
            throw new Error(`Not implemented:${stmt.argument.type}`);
        }
      } else {
        throw new Error('Not implemented');
      }
    }
    /*
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
    */
    // Declaration
    case 'FunctionDeclaration': {
      if (stmt.id?.name == 'smartContract') {
        const ops = compile(stmt.body.body, typeEnv);
        console.error("module {");
        console.error("func.func @smart_contract(%parameter: !michelson.mutez, %storage: !michelson.mutez) -> !michelson.pair<!michelson.list<!michelson.operation>, !michelson.mutez> {");
        console.error(ops.map((op) => op.toString()).join('\n'));
        console.error(" }");
        console.error("}");
        return { op: null, typeEnv };
      } else if (stmt.id?.name == 'MichelsonGetAmount') {
        return { op: null, typeEnv };
      } else if (stmt.id?.name == 'MichelsonMakePair') {
        return { op: null, typeEnv };
      } else if (stmt.id?.name == 'MichelsonMakeOperationList') {
        return { op: null, typeEnv };
      } else if (stmt.id?.name == 'MichelsonMakeResultPair') {
        return { op: null, typeEnv };
      } else {
        return { op: null, typeEnv };
      }
    }
    case 'VariableDeclaration': {
      switch (stmt.kind) {
        case 'var':
          throw new Error('Not implemented');
        case 'let':
          throw new Error('Not implemented');
        case 'const':
          const decl = stmt.declarations[0];
          const callExpression = decl.init as estree.CallExpression;
          const calleeIdentifier = callExpression.callee as estree.Identifier;
          const id = decl.id as estree.Identifier;
          if (calleeIdentifier.name == 'MichelsonMakeResultPair') {
            const fst = callExpression.arguments[0] as estree.Identifier;
            const snd = callExpression.arguments[1] as estree.Identifier;

            const fstType = typeEnv.get(fst.name) as mlir.TypeAttr;
            const sndType = typeEnv.get(snd.name) as mlir.TypeAttr;

            const op = new mlirmichelson.MakePair(
              id.name,
              fst.name,
              snd.name,
              fstType,
              sndType,
            );

            typeEnv.set(id.name, mlirmichelson.PairType(fstType, sndType));

            console.log(op.toString());
            return { op, typeEnv };
          } else if (calleeIdentifier.name == 'MichelsonMakeOperationList') {
            const op = new mlirmichelson.MakeList(
              id.name,
              mlirmichelson.OperationType,
            );

            console.log(op.toString());

            typeEnv.set(
              id.name,
              mlirmichelson.ListType(mlirmichelson.OperationType),
            );
            return { op, typeEnv };
          } else if (calleeIdentifier.name == 'MichelsonGetAmount') {
            const op = new mlirmichelson.GetAmount(id.name);
            console.log(op.toString());
            typeEnv.set(id.name, mlirmichelson.MutezType);
            return { op, typeEnv };
          } else {
            throw new Error('Not implemented');
          }
      }
    }
    /*
    case 'ClassDeclaration':
      throw new Error('Not implemented:ClassDeclaration');
    */
  }
  throw new Error(`Not implemented:${stmt}`);
}

export function compile(
  stmts: (estree.Directive | estree.Statement | estree.ModuleDeclaration)[],
  typeEnv: TypeEnv,
): Op[] {
  const mlirStms: Op[] = [];
  for (const stmt of stmts) {
    const { op, typeEnv: newTypeEnv } = compileStatement(stmt, typeEnv);
    if (op) {
      mlirStms.push(op);
    }
    typeEnv = newTypeEnv;
  }
  return mlirStms;
}
/*
module {
  func.func @smart_contract(%parameter: !michelson.unit, %storage: !michelson.unit)
    -> !michelson.pair<!michelson.list<!michelson.operation>, !michelson.unit> {
  }
}
*/
