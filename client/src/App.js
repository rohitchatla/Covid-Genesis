import logo from "./logo.svg";
import "./App.css";

import React from "react";
import ReactDOM from "react-dom";
import { Provider } from "react-redux";

import { BrowserRouter as Router, Route, Switch } from "react-router-dom";

import IndexHome from "./components/home";
import timeSeries from "./components/timeseries";
import jnb from "./components/jnb";
import sym from "./components/sym";
import xray from "./components/xray";

function App() {
  return (
    <Router>
      <div>
        <div className="container" id="content">
          <Switch>
            <Route exact path="/home" component={IndexHome} />
          </Switch>
          <Switch>
            <Route exact path="/timeseries" component={timeSeries} />
          </Switch>
          <Switch>
            <Route exact path="/xray" component={xray} />
          </Switch>
          <Switch>
            <Route exact path="/symptomanalysis" component={sym} />
          </Switch>
          <Switch>
            <Route exact path="/jnb" component={jnb} />
          </Switch>
        </div>
      </div>
    </Router>
  );
}

export default App;
