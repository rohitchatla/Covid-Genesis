import React, { Component } from "react";
import { Link } from "react-router-dom";
import axios from "../services/Axios";
// components



class home extends Component {
    constructor(props) {
        super(props);
        this.state = {
          work: [],
        };
      }
    async componentDidMount() {
        console.log("hi");
        await axios
          .get(`/ss`) // axios returns a promise
          .then((response) => {
                console.log(response.data);
          })
          .catch(({ response }) => {
              console.log(response);
          });
    };

      render(){
        return (
            <>
              <div>
                <h2>Hi</h2>
              </div>
            </>
          );
      }
  
};

export default home;