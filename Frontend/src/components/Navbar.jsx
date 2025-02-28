import { Link, useLocation } from "react-router-dom";


import { Fragment, useState } from "react";
import { styles } from "../styles.js";
import { Menu, RadioGroup, Transition } from "@headlessui/react";

const navLinks = [
  { id: "", title: "Data Entry" },
  { id: "compliance-category", title: "Compliance Category" },
  { id: "flow-chart", title: "Taxonomy Visualization" },
  { id: "compliance-charts", title: "Compliance Charts" },
];

const Navbar = () => {
  const [active, setActive] = useState("");
  const location = useLocation();
  return (
    <nav
      className={`${styles.paddingX} w-full flex items-center py-5 fixed top-0 z-20  dark:bg-background-dark opacity-80`}
    >
      <div className="w-full flex justify-between items-center max-w-7xl mx-auto">
        {/* Logo and Home Link */}
        <a
          href="/"
          className="flex items-center gap-2"
          onClick={() => {
            window.scrollTo(0, 0);
            setActive("");
          }}
        >
          <p className="text-[18px] font-bold cursor-pointer  dark:text-text-dark">
            Compliance Taxonomy
          </p>
        </a>

        <div className="flex gap-10">
          <ul className=" list-none hidden md:flex flex-row gap-10">
            {navLinks.map((Links) => {
              return (
                <li
                  key={Links.id}
                  className={`${active === Links.title
                    ? "text-[#915eff]"
                    : "dark:text-text-dark"
                    } hover:text-[#915eff] text-[18px] font-medium cursor-pointer`}
                  onClick={() => setActive(Links.title)}
                >
                  <a href={`/${Links.id}`}>{Links.title}</a>
                </li>
              );
            })}
          </ul>

          <div className="md:hidden flex flex-1 justify-end items-center">
            <Menu as="div" className="relative inline-block text-left">
              <Transition
                as={Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-50 mt-2 w-32 origin-top-right divide-y divide-gray-100 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none dark:bg-gray-800">
                  <RadioGroup
                    onChange={(value) => {
                      console.log(value);
                    }}
                  >
                    <div className="p-1">
                      {navLinks.map((Links) => {
                        return (
                          <RadioGroup.Option
                            value={Links.title}
                            key={Links.title}
                          >
                            <Menu.Item>
                              <button
                                className={`group flex w-full items-center rounded-md px-2 py-2  ${location.pathname === Links.id
                                  ? "text-[#915eff]"
                                  : "text-white "
                                  } hover:text-[#915eff]`}
                              >
                                <a href={`/${Links.id}`}>{Links.title}</a>
                              </button>
                            </Menu.Item>
                          </RadioGroup.Option>
                        );
                      })}
                    </div>
                  </RadioGroup>
                </Menu.Items>
              </Transition>
            </Menu>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
