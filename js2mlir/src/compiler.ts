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

function compile(
  stmts: (estree.Directive | estree.Statement | estree.ModuleDeclaration)[],
) {}
