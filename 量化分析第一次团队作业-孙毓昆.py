<!DOCTYPE HTML>
<html>

<head>
    <meta charset="utf-8">

    <title>Jupyter Notebook</title>
    <link rel="shortcut icon" type="image/x-icon" href="/static/base/images/favicon.ico?v=97c6417ed01bdc0ae3ef32ae4894fd03">
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <link rel="stylesheet" href="/static/components/jquery-ui/themes/smoothness/jquery-ui.min.css?v=9b2c8d3489227115310662a343fce11c" type="text/css" />
    <link rel="stylesheet" href="/static/components/jquery-typeahead/dist/jquery.typeahead.min.css?v=7afb461de36accb1aa133a1710f5bc56" type="text/css" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    

    <link rel="stylesheet" href="/static/style/style.min.css?v=29c09309dd70e7fe93378815e5f022ae" type="text/css"/>
    
<link rel="stylesheet" href="/static/auth/css/override.css?v=19ec59d2c4f1203c49fe47462028cd9a" type="text/css" />

    <link rel="stylesheet" href="/custom/custom.css" type="text/css" />
    <script src="/static/components/es6-promise/promise.min.js?v=f004a16cb856e0ff11781d01ec5ca8fe" type="text/javascript" charset="utf-8"></script>
    <script src="/static/components/preact/index.js?v=5b98fce8b86ce059de89f9e728e16957" type="text/javascript"></script>
    <script src="/static/components/proptypes/index.js?v=c40890eb04df9811fcc4d47e53a29604" type="text/javascript"></script>
    <script src="/static/components/preact-compat/index.js?v=d376eb109a00b9b2e8c0d30782eb6df7" type="text/javascript"></script>
    <script src="/static/components/requirejs/require.js?v=6da8be361b9ee26c5e721e76c6d4afce" type="text/javascript" charset="utf-8"></script>
    <script>
      require.config({
          
          urlArgs: "v=20200814211948",
          
          baseUrl: '/static/',
          paths: {
            'auth/js/main': 'auth/js/main.min',
            custom : '/custom',
            nbextensions : '/nbextensions',
            kernelspecs : '/kernelspecs',
            underscore : 'components/underscore/underscore-min',
            backbone : 'components/backbone/backbone-min',
            jquery: 'components/jquery/jquery.min',
            bootstrap: 'components/bootstrap/js/bootstrap.min',
            bootstraptour: 'components/bootstrap-tour/build/js/bootstrap-tour.min',
            'jquery-ui': 'components/jquery-ui/ui/minified/jquery-ui.min',
            moment: 'components/moment/moment',
            codemirror: 'components/codemirror',
            termjs: 'components/xterm.js/dist/xterm',
            typeahead: 'components/jquery-typeahead/dist/jquery.typeahead.min',
          },
          map: { // for backward compatibility
              "*": {
                  "jqueryui": "jquery-ui",
              }
          },
          shim: {
            typeahead: {
              deps: ["jquery"],
              exports: "typeahead"
            },
            underscore: {
              exports: '_'
            },
            backbone: {
              deps: ["underscore", "jquery"],
              exports: "Backbone"
            },
            bootstrap: {
              deps: ["jquery"],
              exports: "bootstrap"
            },
            bootstraptour: {
              deps: ["bootstrap"],
              exports: "Tour"
            },
            "jquery-ui": {
              deps: ["jquery"],
              exports: "$"
            }
          },
          waitSeconds: 30,
      });

      require.config({
          map: {
              '*':{
                'contents': 'services/contents',
              }
          }
      });

      // error-catching custom.js shim.
      define("custom", function (require, exports, module) {
          try {
              var custom = require('custom/custom');
              console.debug('loaded custom.js');
              return custom;
          } catch (e) {
              console.error("error loading custom.js", e);
              return {};
          }
      })
    </script>

    
    

</head>

<body class=""
 
  
 
dir="ltr">

<noscript>
    <div id='noscript'>
      Jupyter Notebook requires JavaScript.<br>
      Please enable it to proceed.
  </div>
</noscript>

<div id="header">
  <div id="header-container" class="container">
  <div id="ipython_notebook" class="nav navbar-brand pull-left"><a href="/tree" title='dashboard'><img src='/static/base/images/logo.png?v=641991992878ee24c6f3826e81054a0f' alt='Jupyter Notebook'/></a></div>

  
  
  


  

  
  
  </div>
  <div class="header-bar"></div>

  
  
</div>

<div id="site">


<div id="ipython-main-app" class="container">

    
    
    <div class="row">
    <div class="navbar col-sm-8">
      <div class="navbar-inner">
        <div class="container">
          <div class="center-nav">
            <p class="navbar-text nav">Password or token:</p>
            <form action="/login?next=%2Fnbconvert%2Fscript%2FDesktop%2F%25E6%2596%25B0%25E5%25BB%25BA%25E6%2596%2587%25E4%25BB%25B6%25E5%25A4%25B9%2F%25E9%2587%258F%25E5%258C%2596%25E5%2588%2586%25E6%259E%2590%25E7%25AC%25AC%25E4%25B8%2580%25E6%25AC%25A1%25E5%259B%25A2%25E9%2598%259F%25E4%25BD%259C%25E4%25B8%259A-%25E5%25AD%2599%25E6%25AF%2593%25E6%2598%2586.ipynb%3Fdownload%3Dtrue" method="post" class="navbar-form pull-left">
              <input type="hidden" name="_xsrf" value="2|8914b08c|19e7ebeecbe8b142763b19be30c867b6|1597300165"/>
              <input type="password" name="password" id="password_input" class="form-control">
              <button type="submit" id="login_submit">Log in</button>
            </form>
          </div>
        </div>
      </div>
    </div>
    </div>
    
    
    
    
    <div class="col-sm-6 col-sm-offset-3 text-left rendered_html">
      <h3>
        Token authentication is enabled
      </h3>
      <p>
        If no password has been configured, you need to open the notebook
        server with its login token in the URL, or paste it above.
        This requirement will be lifted if you
        <b><a href='https://jupyter-notebook.readthedocs.io/en/stable/public_server.html'>
            enable a password</a></b>.
      </p>
      <p>
        The command:
        <pre>jupyter notebook list</pre>
        will show you the URLs of running servers with their tokens,
        which you can copy and paste into your browser. For example:
      </p>
      <pre>Currently running servers:
http://localhost:8888/?token=c8de56fa... :: /Users/you/notebooks
</pre>
      <p>
        or you can paste just the token value into the password field on this
        page.
      </p>
      <p>
        See
        <b><a
         href='https://jupyter-notebook.readthedocs.io/en/stable/public_server.html'>
                the documentation on how to enable a password</a>
        </b>
        in place of token authentication,
        if you would like to avoid dealing with random tokens.
      </p>
      <p>
        Cookies are required for authenticated access to notebooks.
      </p>

    </div>
    
    
</div>


</div>








<script type="text/javascript">
  require(["auth/js/main"], function (auth) {
    auth.login_main();
  });
</script>



<script type='text/javascript'>
  function _remove_token_from_url() {
    if (window.location.search.length <= 1) {
      return;
    }
    var search_parameters = window.location.search.slice(1).split('&');
    for (var i = 0; i < search_parameters.length; i++) {
      if (search_parameters[i].split('=')[0] === 'token') {
        // remote token from search parameters
        search_parameters.splice(i, 1);
        var new_search = '';
        if (search_parameters.length) {
          new_search = '?' + search_parameters.join('&');
        }
        var new_url = window.location.origin + 
                      window.location.pathname + 
                      new_search + 
                      window.location.hash;
        window.history.replaceState({}, "", new_url);
        return;
      }
    }
  }
  _remove_token_from_url();
</script>
</body>

</html>