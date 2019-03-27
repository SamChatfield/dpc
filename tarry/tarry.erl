-module(tarry).
-export([start/0, node_main/2]).

%%
%% MAIN PROCESS
%%

start() ->
    {Initiator, Topology} = read_input(),
    io:format("Initiator: ~p~n", [Initiator]),
    io:format("Topology: ~p~n", [Topology]),
    Nodes = spawn_nodes(Topology),
    io:format("Nodes: ~p~n", [Nodes]),
    send_connections(Topology, Nodes),
    io:format("~nMAIN DONE~n").

read_input() ->
    [Initiator|_] = read_input_line(),
    Topology = read_topology([]),
    {Initiator, Topology}.

read_input_line() ->
    case io:get_line("") of
        eof ->
            eof;
        Data ->
            string:tokens(string:strip(Data, right, $\n), " ")
    end.

read_topology(Topology) ->
    case read_input_line() of
        eof ->
            Topology;
        [Node|ConnectedNodes] ->
            read_topology([{Node, ConnectedNodes}|Topology])
    end.

spawn_nodes(Topology) ->
    [spawn_node(NodeName) || {NodeName, _} <- Topology].

spawn_node(NodeName) ->
    Pid = spawn(tarry, node_main, [self(), NodeName]),
    {NodeName, Pid}.

send_connections([], _) ->
    ok;

send_connections([{NodeName, ConnectedNodeNames}|Topology], Nodes) ->
    {_, NodePid} = lists:keyfind(NodeName, 1, Nodes),
    ConnectedNodes = [lists:keyfind(N, 1, Nodes) || N <- ConnectedNodeNames],
    NodePid ! {connections, ConnectedNodes},
    receive
        {ack} ->
            send_connections(Topology, Nodes)
    end.

%%
%% NODE PROCESS
%%

node_main(MainPid, NodeName) ->
    io:format("Spawned node ~p (~p) pointing back to main (~p)~n", [NodeName, self(), MainPid]),
    node_loop(MainPid, NodeName).

node_loop(MainPid, NodeName) ->
    receive
        {connections, ConnectedNodes} ->
            io:format("Node ~p (~p) is connected to: ~p~n", [NodeName, self(), ConnectedNodes]),
            MainPid ! {ack},
            node_loop(MainPid, NodeName)
    end.
