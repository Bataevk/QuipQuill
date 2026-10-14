// ADR 25 (spike 3.1b): Metro rejects `import(<non-literal>)`, and LangChain's initChatModel
// (pulled in by createAgent) has `import(config.package)`. We always pass a model
// instance, so that branch is dead code: turn it into a rejected promise at build time.
function rejectDynamicImport({ types: t }) {
  return {
    name: 'reject-dynamic-import',
    visitor: {
      CallExpression(path) {
        const [arg] = path.node.arguments;
        if (path.node.callee.type !== 'Import' || !arg || t.isStringLiteral(arg)) return;
        if (t.isTemplateLiteral(arg) && arg.expressions.length === 0) return;
        path.replaceWith(
          t.callExpression(t.memberExpression(t.identifier('Promise'), t.identifier('reject')), [
            t.newExpression(t.identifier('Error'), [
              t.stringLiteral('Dynamic import() is not supported in the React Native bundle'),
            ]),
          ]),
        );
      },
    },
  };
}

module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [rejectDynamicImport],
  };
};
