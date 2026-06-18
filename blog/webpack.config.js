const path = require("path")
const webpack = require("webpack")
const HtmlWebpackPlugin = require('html-webpack-plugin')
const CopyWebpackPlugin = require('copy-webpack-plugin')
const HtmlReplaceWebpackPlugin = require('html-replace-webpack-plugin')

module.exports = {
  entry: {
    "index": "./src/index.js",
  },
  resolve: {
    extensions: [ ".js", ".html", ".json" ]
  },
  output: {
    path: path.join(__dirname, "public"),
    filename: "[name].bundle.js",
    chunkFilename: "[name].[id].js"
  },
  module: {
    rules: [
      {
        test: /\.(html|js)$/,
        exclude: /node_modules/,
        loader: "babel-loader",
        options: {
          presets: ["@babel/preset-env"]
        }
      },
      {
        test: /\.svg$/,
        exclude: /node_modules/,
        loader: 'svg-inline-loader',
        options: {
          removeSVGTagAttrs: true,
          removingTagAttrs: ["font-family"]
        }
      },
      {
        test: /\.(png|jpg|jpeg|gif)$/,
        exclude: /node_modules/,
        loader: 'file-loader',
        options: {
          outputPath: 'images/'
        }
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: "./src/index.ejs",
      filename: "index.html",
      chunks: ["index"]
    }),
    new CopyWebpackPlugin({
      patterns: [
            { from: 'static/' },
            { from: 'src/assets/', to: 'assets/' },
            { from: 'src/diagrams/', to: 'diagrams/' }
      ]
    }),
    new HtmlReplaceWebpackPlugin([
      {
        pattern: /<d-cite\s+key="([^"]*)"\s*\/>/g,
        replacement: (_, key) => '<d-cite key="' + key + '"></d-cite>'
      },
      {
        pattern: '<script type="text/javascript" src="index.bundle.js"></script>',
        replacement: ''
      }
    ])
  ],
  devServer: {
    static: {
      directory: path.join(__dirname, 'public')
    },
    historyApiFallback: true,
    hot: true,
    client: {
      overlay: true
    },
    devMiddleware: {
      stats: "minimal"
    }
  },
  devtool: "inline-source-map"
}